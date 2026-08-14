import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

const CORE_VERTEX_SHADER = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float uTime;
  uniform float uAmplitude;

  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec3 pos = position;
    // Slow swell that is always present, plus a faster ripple that only
    // shows up with audio — the second one is what reads as "talking"
    // rather than "breathing", because its rate changes with the sound
    // instead of running on a fixed clock.
    float swell = sin(pos.x * 4.0 + uTime * 1.5) * sin(pos.y * 4.0 + uTime * 1.2) * 0.04;
    float ripple = sin(pos.y * 11.0 - uTime * 9.0 + uAmplitude * 6.0)
                 * cos(pos.z * 9.0 + uTime * 5.0) * 0.028 * uAmplitude;
    pos += normal * (swell * (0.3 + uAmplitude * 1.8) + ripple);
    vPosition = pos;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;

const CORE_FRAGMENT_SHADER = `
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float uTime;
  uniform float uAmplitude;
  uniform vec3 uColor;

  void main() {
    vec3 viewDir = normalize(cameraPosition - vPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.2);
    float pulse = 0.6 + 0.4 * sin(uTime * 2.0);
    float glow = fresnel * (0.7 + uAmplitude * 2.6) + pulse * 0.15;
    vec3 color = uColor * (0.5 + glow);
    gl_FragColor = vec4(color, fresnel * 0.85 + 0.15);
  }
`;

export type OrbStatus = "idle" | "listening" | "thinking" | "speaking" | "error";

const STATUS_COLORS: Record<OrbStatus, THREE.Color> = {
  idle: new THREE.Color(0x3ac8ff),
  listening: new THREE.Color(0x3ac8ff),
  thinking: new THREE.Color(0xd946ef), // distinct from the gold HUD chrome
  speaking: new THREE.Color(0x37e0c4),
  error: new THREE.Color(0xff4d4d),
};

// A guaranteed motion floor per status, independent of live amplitude —
// real speech amplitude drops near-zero in the brief gaps between words,
// which made "speaking" look like it paused/went idle mid-sentence even
// though audio was still playing. This gets layered on as a slow breathing
// pulse (see `baseline` in tick()) so speaking always reads as visibly
// "alive," with real amplitude spikes still adding on top of it.
const STATUS_ENERGY: Record<OrbStatus, number> = {
  idle: 0,
  listening: 0.06,
  thinking: 0.1,
  speaking: 0.22,
  error: 0.08,
};

export class Orb {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private core: THREE.Mesh;
  private coreMaterial: THREE.ShaderMaterial;
  private wireframe: THREE.Mesh;
  private particles: THREE.Points;
  private clock = new THREE.Clock();

  private targetAmplitude = 0;
  private smoothedAmplitude = 0;
  private targetStatusEnergy = 0;
  private smoothedStatusEnergy = 0;
  private targetColor = STATUS_COLORS.idle.clone();
  private renderedColor = STATUS_COLORS.idle.clone();

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // alpha:true only enables alpha support — clearAlpha still defaults to
    // 1 (opaque) unless set explicitly. Without this, the canvas clears to
    // solid black every frame, hiding every CSS layer behind it (grid,
    // vignette, data-stream columns) — found by testing directly (set body
    // background to red, confirmed nothing showed through anywhere).
    this.renderer.setClearColor(0x000000, 0);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    // Pulled back so the core reads as a compact glowing centre inside the
    // SVG reticle rather than swamping it — the reference gives the rings
    // the visual weight and keeps the core tight.
    this.camera.position.set(0, 0, 9.4);

    this.coreMaterial = new THREE.ShaderMaterial({
      vertexShader: CORE_VERTEX_SHADER,
      fragmentShader: CORE_FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uAmplitude: { value: 0 },
        uColor: { value: STATUS_COLORS.idle.clone() },
      },
      transparent: true,
    });
    this.core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.4, 6), this.coreMaterial);
    this.scene.add(this.core);

    this.wireframe = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.75, 1),
      new THREE.MeshBasicMaterial({ color: 0x00d4ff, wireframe: true, transparent: true, opacity: 0.14 })
    );
    this.scene.add(this.wireframe);

    this.particles = this.buildParticleRing();
    this.scene.add(this.particles);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    // Lower strength than before so the core glows without washing out the
    // fine SVG reticle detail layered over it.
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 1.05, 0.55, 0.2);
    this.composer.addPass(bloom);

    this.resize();
    window.addEventListener("resize", () => this.resize());

    this.renderer.setAnimationLoop(() => this.tick());
  }

  private buildParticleRing(): THREE.Points {
    const count = 900;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 2.3 + Math.random() * 1.2;
      const height = (Math.random() - 0.5) * 0.6;
      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = height;
      positions[i * 3 + 2] = Math.sin(angle) * radius;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0x2ab6ff,
      size: 0.02,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true,
    });
    return new THREE.Points(geometry, material);
  }

  private resize(): void {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private tick(): void {
    const t = this.clock.getElapsedTime();

    // Smooth toward the target so audio-driven jumps don't look jittery.
    this.smoothedAmplitude += (this.targetAmplitude - this.smoothedAmplitude) * 0.25;
    this.smoothedStatusEnergy += (this.targetStatusEnergy - this.smoothedStatusEnergy) * 0.05;

    // Slow breathing pulse scaled by the current status's energy floor —
    // guarantees visible motion while speaking even in the near-silent
    // gaps between words, layered under real amplitude spikes rather than
    // replacing them.
    const breath = 0.55 + 0.45 * Math.sin(t * 3.2);
    const baseline = this.smoothedStatusEnergy * breath;
    const drive = Math.max(this.smoothedAmplitude, baseline);

    this.coreMaterial.uniforms.uTime.value = t;
    this.coreMaterial.uniforms.uAmplitude.value = drive;

    this.renderedColor.copy(this.targetColor);
    (this.coreMaterial.uniforms.uColor.value as THREE.Color).lerp(this.renderedColor, 0.05);

    const scale = 1 + drive * 0.4 + Math.sin(t * 1.4) * 0.02;
    this.core.scale.setScalar(scale);
    this.wireframe.scale.setScalar(1 + drive * 0.22);
    this.wireframe.rotation.y += 0.0025 + drive * 0.014;
    this.wireframe.rotation.x += 0.0012;

    this.particles.rotation.y += 0.0008 + drive * 0.009;
    this.particles.rotation.x = Math.sin(t * 0.12) * 0.02;
    this.camera.position.x = Math.sin(t * 0.08) * 0.025;
    this.camera.position.y = Math.cos(t * 0.07) * 0.02;
    (this.particles.material as THREE.PointsMaterial).opacity = 0.4 + drive * 0.55;

    this.composer.render();
  }

  /** amplitude in [0, 1] — drive this from Web Audio analyser data each frame. */
  setAmplitude(amplitude: number): void {
    this.targetAmplitude = Math.max(0, Math.min(1, amplitude));
  }

  setStatus(status: OrbStatus): void {
    this.targetColor = STATUS_COLORS[status].clone();
    this.targetStatusEnergy = STATUS_ENERGY[status];
  }
}
