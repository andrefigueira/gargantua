# Gargantua

A real-time, ray-marched Schwarzschild black hole rendered in three.js. **Zero textures, zero external assets** — gravitational lensing, the accretion disk, dust volumetrics, the starfield and the nebula are all generated per-pixel from physics and hash noise.

The whole effect is a single self-contained drop-in module — `src/BlackHole.js`.

## Try it

```bash
npm install
npm run dev
```

Open `http://localhost:5173/`. Click-drag to orbit, scroll to zoom. Stop touching the mouse and a slow auto-orbit takes over.

## Use it in your own project

`src/BlackHole.js` has no dependencies on the demo. Drop it into any three.js project that uses `EffectComposer`:

```js
import { createBlackHole } from './BlackHole.js';

const bh = createBlackHole({
  diskInner:    2.6,    // Schwarzschild radii
  diskOuter:    9.0,
  exposure:     1.0,
  dustOpacity:  3.2,
  dustEmission: 1.10,
});

bh.setSize(window.innerWidth, window.innerHeight);
composer.addPass(bh.pass);                // must be the FIRST pass
composer.addPass(bloomPass);              // then your bloom
composer.addPass(new OutputPass());

function frame(nowMs) {
  bh.update(camera, controls.target, nowMs * 0.001);
  composer.render();
  requestAnimationFrame(frame);
}
```

That's the whole integration.

## What's modelled

- **Light bending.** Photon geodesics are integrated using a symplectic Euler step on `a = −1.5 · h² · r̂ / r⁴`, the 3-vector form of the Schwarzschild null-geodesic equation. `h² = |r × v|²` is conserved exactly along each ray.
- **The shadow is analytic.** A photon is captured iff its impact parameter `b = |L| < 3√3/2 · RS`. The silhouette is computed from this analytic test, not the numerical integration, so it stays a perfectly smooth circle at every camera angle.
- **Accretion disk.** Geometrically thin (h/r ≈ 0.02) Keplerian disk in the equatorial plane. Material orbits at Ω(r) ∝ r⁻³ᐟ². The disk is sampled volumetrically with stratified sub-sampling and Beer-Lambert front-to-back compositing.
- **Dust.** A two-layer vertical profile — a thin bright slab plus a broader dim halo — gives the disk a glowing volumetric atmosphere above and below the plane.
- **Relativistic effects.** Gravitational redshift on the disk emission, plus a deliberately gentle Doppler boost on the prograde-orbiting side (Nolan suppressed it in *Interstellar* to keep the disk readable from any side — same tradeoff here).
- **Background.** Procedural starfield (cubemap-projected hash) and triplanar-blended nebula. No seams across the sphere.

## Quality knobs

| Option         | Default | What it does                                   |
| -------------- | ------- | ---------------------------------------------- |
| `diskInner`    | `2.6`   | Inner edge of the disk in Schwarzschild radii. |
| `diskOuter`    | `9.0`   | Outer edge.                                    |
| `exposure`     | `1.0`   | Linear multiplier on final emission.           |
| `dustOpacity`  | `3.2`   | Beer-Lambert absorption coefficient of dust.   |
| `dustEmission` | `1.10`  | Brightness scalar for the disk dust glow.      |

Read the file — every magic number has a comment explaining what it does.

## Performance

The fragment shader does up to 180 geodesic steps per pixel with adaptive sub-sampling near the disk plane. On an M-series Mac at 1.5× DPR, expect:

- Apple M1: ~60 fps at 1440p
- Apple M2 / M3 Pro: 60 fps at 4K
- Discrete NVIDIA (RTX 3060+): 60 fps at 4K

If you're hitting the GPU too hard:

- Lower `renderer.setPixelRatio(...)` in `main.js` (1.0 cuts cost ~30% from default).
- Reduce `MAX_STEPS` in `BlackHole.js` (default 180; 120 is still solid).
- Drop `SUBS` in the sub-sampling loop from 8 to 6 (visible only on the thinnest filaments).

## References

- James, Tunzelmann, Franklin, Thorne — *Gravitational Lensing by Spinning Black Holes in Astrophysics, and in the Movie Interstellar*. Class. Quantum Grav. 32 (2015) 065001. The original paper from the *Interstellar* render team.
- Misner, Thorne, Wheeler — *Gravitation*, §25. The null-geodesic equation in Schwarzschild coordinates.
- Riccardo Antonelli — *How to draw a black hole*. The canonical reference for the cartesian-coordinate trick used in the geodesic integration.

## License

MIT. See [LICENSE](LICENSE).
