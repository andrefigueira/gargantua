/**
 * Gargantua — a procedural, ray-marched Schwarzschild black hole for three.js.
 *
 * This is a single drop-in fragment-shader pass. It uses zero textures and
 * zero external assets — gravitational lensing, the accretion disk, dust
 * volumetrics, the starfield and the nebula are all generated per-pixel
 * from physics and hash noise.
 *
 * ─── Physics notes ──────────────────────────────────────────────────────────
 *
 *   • The spacetime is non-rotating Schwarzschild with the event horizon at
 *     r = RS (Schwarzschild radius). Distances throughout the shader are in
 *     units of RS, so RS = 1.
 *
 *   • Photon geodesics are integrated using a symplectic Euler step on the
 *     acceleration  a = -1.5 · h² · r̂ / r⁴,  where h² = |r × v|² is the
 *     squared specific angular momentum, conserved along the geodesic. This
 *     is the standard reduction of the Schwarzschild null-geodesic equation
 *     when expressed in cartesian coordinates of the orbital plane.
 *
 *   • The black-hole shadow is computed analytically rather than from the
 *     numerical integration. A photon is captured iff its impact parameter
 *     b = |L| / |dr/dλ| is below the critical value b_crit = 3√3/2 · RS.
 *     Because b is exactly conserved, this gives a perfectly smooth circular
 *     silhouette in screen space — independent of integration step count.
 *
 *   • The accretion disk is geometrically thin (h/r ≈ 0.02) and lives in the
 *     equatorial plane y = 0. Material orbits at the Keplerian rate
 *     Ω(r) ∝ r^(−3/2). The disk is ray-marched volumetrically with stratified
 *     sub-sampling and Beer-Lambert front-to-back compositing, so dust
 *     filaments correctly emit and absorb the light passing through them.
 *
 *   • Relativistic effects rendered: gravitational light bending (full),
 *     gravitational redshift on disk emission, and a gentle Doppler boost on
 *     the prograde-orbiting side of the disk. The Doppler exponent is
 *     deliberately low so both sides of the disk read as bright — matching
 *     the cinematic Interstellar look rather than the maximally physically
 *     accurate one.
 *
 * ─── References ────────────────────────────────────────────────────────────
 *
 *   James, Tunzelmann, Franklin, Thorne — "Gravitational Lensing by Spinning
 *     Black Holes in Astrophysics, and in the Movie Interstellar."
 *     Class. Quantum Grav. 32 (2015) 065001.
 *   Misner, Thorne, Wheeler — Gravitation, §25 (null geodesics).
 *   Riccardo Antonelli — "How to draw a black hole" (well-known reference
 *     for the cartesian Schwarzschild geodesic integration trick).
 *
 * ─── Usage ─────────────────────────────────────────────────────────────────
 *
 *   import { createBlackHole } from './BlackHole.js';
 *
 *   const bh = createBlackHole();
 *   bh.setSize(window.innerWidth, window.innerHeight);
 *   composer.addPass(bh.pass);          // make this the FIRST pass
 *   // ...add bloom + output passes after...
 *
 *   function animate(now) {
 *     bh.update(camera, controls.target, now * 0.001);
 *     composer.render();
 *   }
 */

import * as THREE from 'three';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

/* ────────────────────────────────────────────────────────────────────────── *
 *  Defaults & types
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * @typedef {Object} BlackHoleOptions
 * @property {number} [diskInner=2.6]   Inner disk edge (Schwarzschild radii).
 * @property {number} [diskOuter=9.0]   Outer disk edge (Schwarzschild radii).
 * @property {number} [exposure=1.0]    Linear multiplier on final emission.
 * @property {number} [dustOpacity=3.2] Beer-Lambert absorption coefficient
 *                                       of the disk dust.
 * @property {number} [dustEmission=1.10] Brightness scalar for the disk dust.
 */

/**
 * @typedef {Object} BlackHole
 * @property {import('three/addons/postprocessing/ShaderPass.js').ShaderPass} pass
 * @property {(camera: THREE.Camera, target: THREE.Vector3, time: number) => void} update
 * @property {(width: number, height: number) => void} setSize
 * @property {Record<string, THREE.IUniform>} uniforms
 */

const DEFAULTS = Object.freeze({
  diskInner:    2.6,
  diskOuter:    9.0,
  exposure:     1.0,
  dustOpacity:  3.2,
  dustEmission: 1.10,
});

/* ────────────────────────────────────────────────────────────────────────── *
 *  Shader source
 *  -------------------------------------------------------------------------
 *  The fragment shader is assembled from logical chunks below so each
 *  section can be read and edited in isolation. Concatenation order matters
 *  because GLSL has no forward declarations — every helper must precede its
 *  first call site.
 * ────────────────────────────────────────────────────────────────────────── */

const VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const GLSL_UNIFORMS = /* glsl */ `
  precision highp float;
  varying vec2 vUv;

  uniform vec2  uResolution;
  uniform float uTime;
  uniform vec3  uCamPos;
  uniform vec3  uCamForward;
  uniform vec3  uCamRight;
  uniform vec3  uCamUp;
  uniform float uDiskInner;
  uniform float uDiskOuter;
  uniform float uExposure;
  uniform float uDustOpacity;
  uniform float uDustEmission;

  // Distances are in units of the Schwarzschild radius (RS = 1).
  const float RS        = 1.0;
  const float FOV       = 1.1;
  const int   MAX_STEPS = 180;

  // 3√3/2 — critical impact parameter for capture by a Schwarzschild BH.
  const float B_CRIT = 2.59807621135;
`;

const GLSL_NOISE = /* glsl */ `
  float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }

  // 4-octave FBM. Used wherever cheap, broad-stroke noise is enough.
  float fbm(vec2 p) {
    const mat2 R = mat2(0.8, -0.6, 0.6, 0.8);
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) {
      v += a * vnoise(p);
      p = R * p * 2.03;
      a *= 0.5;
    }
    return v;
  }

  // 6-octave FBM. Used for disk fractal detail where the extra octaves
  // pay off as visible filamentary structure.
  float fbm6(vec2 p) {
    const mat2 R = mat2(0.8, -0.6, 0.6, 0.8);
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) {
      v += a * vnoise(p);
      p = R * p * 2.03;
      a *= 0.5;
    }
    return v;
  }

  // Ridged variant of FBM. Produces the streaky "vein" structure of hot
  // plasma in long-exposure astrophotography.
  float ridge(vec2 p) {
    return 1.0 - abs(2.0 * fbm(p) - 1.0);
  }
`;

const GLSL_BACKGROUND = /* glsl */ `
  // Stars + nebula sampled from a normalised viewing direction.
  // Stars use a single-face cubemap projection — they're point samples so
  // face boundaries are invisible. The nebula uses a triplanar blend so the
  // continuous noise has no visible cube-face seams.
  vec3 starfield(vec3 dir) {
    vec3 a = abs(dir);
    vec2 uv;
    float maxC = max(a.x, max(a.y, a.z));
    if      (maxC == a.x) uv = dir.yz / a.x;
    else if (maxC == a.y) uv = dir.xz / a.y;
    else                  uv = dir.xy / a.z;

    vec3 col = vec3(0.0);

    // Three star layers at different cell sizes. Larger cells → bigger
    // sub-pixel footprint → no flicker under camera motion.
    for (float layer = 0.0; layer < 3.0; layer += 1.0) {
      float scale  = 90.0 + layer * 130.0;
      float thresh = 0.978 - layer * 0.003;
      vec2 g = uv * scale;
      vec2 id = floor(g), fp = fract(g);
      float h = hash21(id + layer * 37.0);
      if (h > thresh) {
        vec2 starP = vec2(hash21(id + 11.0), hash21(id + 23.0));
        float d = length(fp - starP);
        float core = smoothstep(0.08, 0.0, d);
        float glow = smoothstep(0.22, 0.0, d) * 0.12;
        float tint = hash21(id + 47.0);
        vec3 c = mix(vec3(0.85, 0.95, 1.25), vec3(1.25, 0.95, 0.7), tint);
        col += c * (core + glow) * 0.55;
      }
    }

    // Triplanar nebula — blends three orthogonal planar fbm samples by the
    // direction's per-axis weight. Seamless across the sphere.
    vec3 w = a / max(0.0001, a.x + a.y + a.z);
    float ofs = uTime * 0.005;

    float dustA  = fbm(dir.yz * 1.8 + vec2(ofs,  0.0));
    float dustB  = fbm(dir.xz * 1.8 + vec2(0.0,  ofs));
    float dustC  = fbm(dir.xy * 1.8 - vec2(ofs,  0.0));
    float dust   = dustA * w.x + dustB * w.y + dustC * w.z;

    float dust2A = fbm(dir.yz * 4.0 - vec2(0.0, ofs * 1.6));
    float dust2B = fbm(dir.xz * 4.0 + vec2(ofs * 1.6, 0.0));
    float dust2C = fbm(dir.xy * 4.0 + vec2(0.0, ofs * 1.6));
    float dust2  = dust2A * w.x + dust2B * w.y + dust2C * w.z;

    float band = exp(-pow(dir.y * 1.4, 2.0));
    col += vec3(0.06, 0.04, 0.12) * pow(dust,  1.6) * band;
    col += vec3(0.10, 0.05, 0.04) * pow(dust2, 2.0) * 0.6 * band;
    col += vec3(0.012, 0.012, 0.02);
    return col;
  }
`;

const GLSL_DISK = /* glsl */ `
  // Volumetric sample of the accretion disk at a 3D point.
  // Returns (RGB emission, absorption σ).  σ has units of "per unit length"
  // and is composited via Beer-Lambert in the ray-march below.
  vec4 diskSample(vec3 p, vec3 vel) {
    float r = length(p.xz);
    if (r < uDiskInner - 1.2 || r > uDiskOuter + 1.5) return vec4(0.0);

    // Two-layer vertical density profile:
    //   • a thin bright slab (the disk proper, scale height H ≈ 0.05–0.2)
    //   • a much broader dim halo (the dust extending above and below the
    //     disk plane that gives the Gargantua image its glowing atmosphere)
    float H     = 0.035 + 0.018 * r;
    float Hhalo = 0.30  + 0.10  * r;
    float vert  = exp(-(p.y * p.y) / (H     * H    ));
    float halo  = exp(-(p.y * p.y) / (Hhalo * Hhalo)) * 0.25;
    if (vert + halo < 0.005) return vec4(0.0);

    // Radial profile with soft inner and outer fade-outs.
    float rN      = clamp((r - uDiskInner) / (uDiskOuter - uDiskInner), 0.0, 1.0);
    float edgeIn  = smoothstep(uDiskInner - 0.5, uDiskInner + 0.5, r);
    float edgeOut = 1.0 - smoothstep(uDiskOuter - 2.0, uDiskOuter + 1.0, r);
    float radial  = pow(1.0 - rN, 1.2) * edgeIn * edgeOut;
    if (radial < 0.001) return vec4(0.0);

    // Differential Keplerian rotation, sampled in a co-rotating cartesian
    // frame to avoid the φ = ±π seam of polar coordinates.
    float omega  = 2.2 / pow(r, 1.5);
    float rotAng = uTime * omega * 2.4;
    float cs = cos(rotAng), sn = sin(rotAng);
    vec2  q  = vec2(cs * p.x - sn * p.z, sn * p.x + cs * p.z);

    // Fractal disk structure — compounded multiplicatively so high-detail
    // variations ride on top of large-scale turbulence.
    float n1     = fbm6(q * 0.55);
    float n2     = fbm6(q * 1.55);
    float detail = fbm (q * 5.5);
    float fine   = fbm (q * 12.0);
    float lane   = fbm (q * 3.4);
    float veins  = ridge(q * 2.2);

    float clumps = (0.55 + 0.95 * n1 + 0.50 * n2)
                 * (0.75 + 0.45 * detail)
                 * (0.85 + 0.30 * fine)
                 * mix(0.85, 1.15, veins);
    float laneMod     = 0.55 + 0.45 * lane;
    float diskDensity = clumps * laneMod * radial * vert;
    float hazeDensity = (0.6 + 0.4 * n1) * radial * halo;

    // Blackbody-like colour ramp: white-hot inner edge → gold → warm orange
    // → deep red outer, biased toward the Interstellar warm palette.
    vec3 white = vec3(1.85, 1.70, 1.45);
    vec3 gold  = vec3(1.55, 1.10, 0.55);
    vec3 warm  = vec3(1.30, 0.55, 0.18);
    vec3 deep  = vec3(0.55, 0.18, 0.06);
    vec3 baseC = mix(white, gold, smoothstep(0.0,  0.22, rN));
    baseC      = mix(baseC, warm, smoothstep(0.22, 0.55, rN));
    baseC      = mix(baseC, deep, smoothstep(0.55, 1.00, rN));

    // View-independent ISCO ring — the searing white halo around the
    // horizon visible from any viewing angle. Driven by radial position
    // alone so it doesn't disappear when looking edge-on.
    float ringR        = smoothstep(uDiskInner + 2.0, uDiskInner - 0.2, r);
    float ringIntensity = pow(ringR, 1.8);
    vec3  ringColor    = vec3(1.55, 1.40, 1.15);

    // Gentle Doppler beaming — present but not dominating, matching the
    // suppressed beaming of the Interstellar render.
    vec3  tangent = normalize(vec3(-p.z, 0.0, p.x));
    float vOrb    = min(0.55, 0.95 / sqrt(max(r, 1.2)));
    float beta    = clamp(dot(tangent * vOrb, normalize(-vel)), -0.9, 0.9);
    float doppler = clamp(1.0 / (1.0 - beta), 0.7, 1.7);
    float gravRed = sqrt(max(0.0, 1.0 - RS / max(r, RS + 0.01)));

    // Final emission: thin-disk plasma + view-independent ring + warm haze.
    vec3 emit  = baseC * diskDensity * pow(doppler, 1.1) * gravRed * uDustEmission;
         emit += ringColor * ringIntensity
                  * (vert * (0.6 + 0.4 * clumps) + halo * 0.4) * gravRed;
         emit += vec3(0.55, 0.30, 0.14) * hazeDensity * (0.45 + 0.55 * n2);

    float sigma = (diskDensity + hazeDensity * 0.4) * uDustOpacity;
    return vec4(emit, sigma);
  }
`;

const GLSL_MAIN = /* glsl */ `
  // Per-pixel ray direction from the camera basis uniforms.
  vec3 rayDirection(vec2 ndc) {
    float aspect = uResolution.x / uResolution.y;
    float t = tan(FOV * 0.5);
    return normalize(
        uCamForward
      + uCamRight * (ndc.x * aspect * t)
      + uCamUp    * (ndc.y          * t)
    );
  }

  void main() {
    vec2 ndc = vUv * 2.0 - 1.0;
    vec3 vel = rayDirection(ndc);

    // Per-pixel deterministic jitter. Phase-shifts neighbouring rays so the
    // discrete disk-slab sub-samples don't all hit or all miss the gaussian
    // peak together. Spatial-only — no time term, so it never flickers.
    float pxJit = hash21(gl_FragCoord.xy);
    vec3  pos   = uCamPos + vel * pxJit * 0.5;

    // Conserved specific angular momentum about the BH (Schwarzschild is
    // a central-potential metric — L is exactly conserved on null geodesics).
    vec3  L  = cross(pos, vel);
    float h2 = dot(L, L);

    // Analytic shadow test. Decoupled from the numerical integration so the
    // silhouette is a perfect circle at every camera angle.
    bool inShadow = sqrt(h2) < B_CRIT;

    // Escape radius scales with camera distance so the simulation domain
    // always contains both camera and BH at any zoom level.
    float camR    = length(uCamPos);
    float escapeR = max(50.0, camR * 1.6);
    float esc2    = escapeR * escapeR;

    vec3  emission     = vec3(0.0);
    float transmittance = 1.0;

    for (int i = 0; i < MAX_STEPS; i++) {
      float r2 = dot(pos, pos);
      float r  = sqrt(r2);

      // Adaptive step — fine near the BH, coarse far away, but capped so
      // far-field strides can never jump over the thin disk slab.
      float dt = clamp(0.04 + r * 0.18, 0.04, 0.9);

      // Schwarzschild null-geodesic acceleration. Derived from the radial
      // equation d²u/dφ² + u = 3 M u² with u = 1/r, re-expressed in 3-vector
      // form using h² = |r × v|².
      //   r⁻⁵ = (r²)⁻² · r⁻¹  →  cheaper than pow(r², 2.5) on most GPUs.
      float invR2 = 1.0 / r2;
      vec3  acc   = -1.5 * h2 * pos * (invR2 * invR2 * inversesqrt(r2));
      vel        += acc * dt;
      vec3 newPos = pos + vel * dt;

      // Cheap conservative test for whether the segment skirts the disk
      // slab. Only sub-sample the disk inside this region — outside it we
      // just take a single geodesic step.
      float rA = length(pos.xz);
      float rB = length(newPos.xz);
      bool radialBand = max(rA, rB) > uDiskInner - 1.2 &&
                        min(rA, rB) < uDiskOuter + 1.5;
      bool ySlab = abs(pos.y)    < 0.45 ||
                   abs(newPos.y) < 0.45 ||
                   sign(pos.y) != sign(newPos.y);

      if (radialBand && ySlab) {
        // Stratified sub-sampling of the disk segment. Per-pixel + per-step
        // jitter randomises sample positions within each stratum, breaking
        // the staircase pattern that fixed offsets produce.
        //
        // SUBS is adaptive — segments that fully cross the slab need more
        // samples to capture the gaussian peak, whereas segments that only
        // skim above or below the slab need fewer. Saves ~30% disk-band
        // cost on average without changing visuals.
        bool fullCross = sign(pos.y) != sign(newPos.y);
        int  SUBS      = fullCross ? 8 : 4;
        float subDt    = dt / float(SUBS);
        float stepJit  = hash21(gl_FragCoord.xy + vec2(float(i) * 1.7, 0.0));
        for (int j = 0; j < 8; j++) {
          if (j >= SUBS) break;
          float t   = (float(j) + stepJit) / float(SUBS);
          vec3 sp   = mix(pos, newPos, t);
          vec4 samp = diskSample(sp, vel);
          if (samp.w > 0.0) {
            float opacity = 1.0 - exp(-samp.w * subDt);
            emission       += samp.rgb * subDt * transmittance;
            transmittance  *= 1.0 - opacity;
          }
        }
      }

      pos = newPos;

      // Numerical capture is only an early-exit optimisation — the analytic
      // b-test above is what actually decides the silhouette.
      if (dot(pos, pos) < RS * RS)  break;
      if (dot(pos, pos) > esc2)     break;
      if (transmittance < 0.005)    break;
    }

    vec3 colour = emission;
    if (!inShadow) {
      colour += starfield(normalize(vel)) * transmittance;
    }

    colour *= uExposure;

    // Subtle vignette — softens screen-edge brightness for a cinematic feel.
    float vig = smoothstep(1.4, 0.4, length(ndc));
    colour *= mix(0.85, 1.0, vig);

    gl_FragColor = vec4(colour, 1.0);
  }
`;

const FRAGMENT_SHADER =
  GLSL_UNIFORMS +
  GLSL_NOISE +
  GLSL_BACKGROUND +
  GLSL_DISK +
  GLSL_MAIN;

/* ────────────────────────────────────────────────────────────────────────── *
 *  Public API
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Build a self-contained black-hole ShaderPass with update + setSize helpers.
 *
 * The returned pass writes the entire image — it ignores any input texture —
 * so it must be the first pass in your EffectComposer chain. Add bloom and
 * the OutputPass after it.
 *
 * @param {BlackHoleOptions} [options]
 * @returns {BlackHole}
 */
export function createBlackHole(options = {}) {
  const opts = { ...DEFAULTS, ...options };

  const pass = new ShaderPass({
    uniforms: {
      // ShaderPass needs this declared; the shader itself ignores it.
      tDiffuse:      { value: null },
      uResolution:   { value: new THREE.Vector2(1, 1) },
      uTime:         { value: 0 },
      uCamPos:       { value: new THREE.Vector3() },
      uCamForward:   { value: new THREE.Vector3(0, 0, -1) },
      uCamRight:     { value: new THREE.Vector3(1, 0, 0) },
      uCamUp:        { value: new THREE.Vector3(0, 1, 0) },
      uDiskInner:    { value: opts.diskInner },
      uDiskOuter:    { value: opts.diskOuter },
      uExposure:     { value: opts.exposure },
      uDustOpacity:  { value: opts.dustOpacity },
      uDustEmission: { value: opts.dustEmission },
    },
    vertexShader:   VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
  });

  // Reused per-frame to avoid allocations in update().
  const forward = new THREE.Vector3();
  const right   = new THREE.Vector3();
  const up      = new THREE.Vector3();
  const WORLD_UP = new THREE.Vector3(0, 1, 0);

  /**
   * Synchronise the shader's camera basis with a three.js camera + target.
   * Call this once per frame before rendering the composer.
   *
   * @param {THREE.Camera} camera
   * @param {THREE.Vector3} target
   * @param {number} time  Seconds since some fixed epoch (drives animation).
   */
  function update(camera, target, time) {
    forward.copy(target).sub(camera.position).normalize();

    // Avoid gimbal lock when looking straight up or straight down.
    if (Math.abs(forward.dot(WORLD_UP)) > 0.999) {
      right.set(1, 0, 0);
    } else {
      right.crossVectors(forward, WORLD_UP).normalize();
    }
    up.crossVectors(right, forward).normalize();

    const u = pass.uniforms;
    u.uTime.value = time;
    u.uCamPos.value.copy(camera.position);
    u.uCamForward.value.copy(forward);
    u.uCamRight.value.copy(right);
    u.uCamUp.value.copy(up);
  }

  /**
   * Update the resolution uniform. Call from your resize handler.
   *
   * @param {number} width
   * @param {number} height
   */
  function setSize(width, height) {
    pass.uniforms.uResolution.value.set(width, height);
  }

  return { pass, update, setSize, uniforms: pass.uniforms };
}
