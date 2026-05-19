/**
 * Gargantua — demo entry point.
 *
 * Wires up a WebGL renderer, an OrbitControls-driven camera, the BlackHole
 * pass, bloom, output, and SMAA. The black-hole module itself lives in
 * BlackHole.js and has no dependencies on anything here — copy it into any
 * three.js project that already has an EffectComposer.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';
import { createBlackHole } from './BlackHole.js';

/* ── Constants ────────────────────────────────────────────────────────────── */

// Renders at most this many device pixels per CSS pixel. A retina cap of 1.5
// is the sweet spot — SMAA + bloom hide the difference between 1.5 and 2,
// and the GPU cost scales linearly with pixel count.
const PIXEL_RATIO_CAP = 1.5;

// How long to wait after the last user interaction before resuming the
// gentle auto-orbit. Long enough that brief mouse-up pauses don't trigger.
const AUTO_ROTATE_DELAY_MS = 2500;

/* ── Renderer ─────────────────────────────────────────────────────────────── */

const canvas = document.getElementById('c');

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,            // SMAA handles AA — MSAA on the BH pass is wasted.
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, PIXEL_RATIO_CAP));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping       = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.85;
renderer.outputColorSpace  = THREE.SRGBColorSpace;

/* ── Camera + OrbitControls ───────────────────────────────────────────────── */

// A real PerspectiveCamera is kept so OrbitControls works as users expect.
// The BlackHole module reads its world transform each frame and converts it
// into shader uniforms.
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(7.5, 2.0, 7.5);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.rotateSpeed   = 0.55;
controls.zoomSpeed     = 0.9;
controls.enablePan     = false;  // panning makes little sense — always orbit the BH.
controls.minDistance   = 3.2;    // keep the camera outside the photon sphere.
controls.maxDistance   = 120;
controls.minPolarAngle = 0.05;
controls.maxPolarAngle = Math.PI - 0.05;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.25;

// Pause auto-orbit while the user is interacting, then resume after a delay.
let lastInputAt = -Infinity;
const onInteract = () => {
  lastInputAt = performance.now();
  controls.autoRotate = false;
};
controls.addEventListener('start',  onInteract);
controls.addEventListener('change', onInteract);

/* ── Post-processing chain ────────────────────────────────────────────────── */

const composer = new EffectComposer(renderer);

const blackHole = createBlackHole();
blackHole.setSize(window.innerWidth, window.innerHeight);
composer.addPass(blackHole.pass);

const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  /* strength  */ 0.85,
  /* radius    */ 0.70,
  /* threshold */ 0.65
);
composer.addPass(bloom);

composer.addPass(new OutputPass());

const smaa = new SMAAPass(
  window.innerWidth  * renderer.getPixelRatio(),
  window.innerHeight * renderer.getPixelRatio()
);
composer.addPass(smaa);

/* ── Frame loop ───────────────────────────────────────────────────────────── */

const startedAt = performance.now();

function render() {
  const now = performance.now();
  const t   = (now - startedAt) * 0.001;

  if (!controls.autoRotate && now - lastInputAt > AUTO_ROTATE_DELAY_MS) {
    controls.autoRotate = true;
  }
  controls.update();
  blackHole.update(camera, controls.target, t);
  composer.render();

  requestAnimationFrame(render);
}

/* ── Resize handling ──────────────────────────────────────────────────────── */

window.addEventListener('resize', () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const dpr = renderer.getPixelRatio();

  renderer.setSize(w, h);
  composer.setSize(w, h);
  blackHole.setSize(w, h);
  bloom.setSize(w, h);
  smaa.setSize(w * dpr, h * dpr);

  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});

render();
