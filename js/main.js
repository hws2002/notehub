import * as THREE from "https://cdn.skypack.dev/three@0.132.2";
import { TWEEN } from "https://cdn.skypack.dev/three@0.132.2/examples/jsm/libs/tween.module.min.js";
import { setupScene } from "./scene.js";
import { createGraph } from "./graph.js";

// --- Initial Setup ---
const { scene, camera, renderer, controls } = setupScene();
const { nodes, edges } = createGraph(scene);

// --- Frame the whole scene ---
// Create a bounding box that will encompass all spheres
const boundingBox = new THREE.Box3();

// Expand the bounding box to include each sphere
nodes.forEach((node) => {
  boundingBox.expandByObject(node.sphere);
});

// Calculate the center and size of the bounding box
const center = new THREE.Vector3();
boundingBox.getCenter(center);
const size = new THREE.Vector3();
boundingBox.getSize(size);

// Adjust camera to fit the bounding box
const maxDim = Math.max(size.x, size.y, size.z);
const fov = camera.fov * (Math.PI / 180);
camera.position.z = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5; // Increased padding for a wider view
camera.position.y = center.y;
camera.position.x = center.x;
controls.target.copy(center);
controls.update();

// --- Interactivity & Physics State ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let selectedNode = null;
let isDragging = false;
let plane = new THREE.Plane();
let intersection = new THREE.Vector3();

// --- Physics Constants ---
const DAMPING = 0.95; // Friction to slow things down
const TIME_STEP = 0.02;

// Animation loop
function animate() {
  requestAnimationFrame(animate);

  TWEEN.update(); // Update animations

  // --- Physics Simulation Step ---
  nodes.forEach((node) => {
    // If the node is not being dragged, apply physics
    if (node !== selectedNode) {
      // Apply force to velocity
      node.velocity.add(node.force.clone().multiplyScalar(TIME_STEP));
      // Apply damping (friction)
      node.velocity.multiplyScalar(DAMPING);
      // Update position
      node.sphere.position.add(node.velocity.clone().multiplyScalar(TIME_STEP));
      // Reset force for the next frame
      node.force.set(0, 0, 0);
    }

    // Reset force for the next frame (for all nodes)
    node.force.set(0, 0, 0);

    // Update the label position
    node.update();
  });

  // Update the lines to follow the moving nodes
  edges.forEach((edge) => {
    const positions = edge.line.geometry.attributes.position;
    positions.setXYZ(
      0,
      edge.source.sphere.position.x,
      edge.source.sphere.position.y,
      edge.source.sphere.position.z
    );
    positions.setXYZ(
      1,
      edge.target.sphere.position.x,
      edge.target.sphere.position.y,
      edge.target.sphere.position.z
    );
    positions.needsUpdate = true;
  });
  controls.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function onMouseMove(event) {
  // Only drag if a node is selected and we are in dragging mode
  if (!selectedNode || !isDragging) return;

  // Update mouse coordinates
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  // Update the intersection point for the drag force calculation
  raycaster.setFromCamera(mouse, camera);
  if (raycaster.ray.intersectPlane(plane, intersection)) {
    // Directly move the node to the cursor position
    // Store the current position before updating it, to calculate velocity later
    selectedNode.previousPosition.copy(selectedNode.sphere.position);
    selectedNode.sphere.position.copy(intersection);
  }
}

function onMouseDown(event) {
  if (event.button !== 0) return; // Only act on left mouse button

  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(nodes.map((n) => n.sphere));

  if (intersects.length > 0) {
    const intersectedObject = intersects[0].object;
    const clickedNode = intersectedObject.userData.node;
    // Select the node on mouse down
    selectNode(clickedNode);
  }
}

function selectNode(node) {
  selectedNode = node;
  // Add emissive glow to the selected sphere
  selectedNode.sphere.material.emissive.setHex(0x555555);
  // Highlight connected edges
  selectedNode.connectedEdges.forEach((edge) => {
    edge.line.material.opacity = 1.0;
    edge.line.material.color.setHex(0xffffff); // Make it bright white
  });

  // Enable dragging for this node
  isDragging = true;
  // Prevent camera movement during drag, but keep controls enabled to track state
  controls.enableRotate = false;
  controls.enablePan = false;

  // Set up the drag plane
  plane.setFromNormalAndCoplanarPoint(
    camera.getWorldDirection(plane.normal),
    selectedNode.sphere.position
  );

  // Store the initial position for velocity calculation on release
  selectedNode.previousPosition.copy(selectedNode.sphere.position);
}

function onMouseUp(event) {
  // Deselect the node when the mouse button is released
  if (selectedNode) {
    deselectNode();
  }
}

function deselectNode() {
  if (!selectedNode) return;
  // Remove emissive glow

  // Calculate and apply release velocity
  const releaseVelocity = new THREE.Vector3().subVectors(
    selectedNode.sphere.position,
    selectedNode.previousPosition
  );
  selectedNode.velocity.copy(releaseVelocity).divideScalar(TIME_STEP); // Scale velocity by time

  selectedNode.sphere.material.emissive.setHex(0x000000);
  // Reset connected edges
  selectedNode.connectedEdges.forEach((edge) => {
    edge.line.material.opacity = 0.5;
    edge.line.material.color.setHex(0xffffff); // Or back to its original if they varied
  });

  // Disable dragging and re-enable camera controls
  isDragging = false;
  // Re-enable camera movement
  controls.enableRotate = true;
  controls.enablePan = true;
  selectedNode = null;
}

function onDoubleClick(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(nodes.map((n) => n.sphere));

  if (intersects.length > 0) {
    // Prevent double-click from triggering selection logic
    deselectNode(); // Always deselect on double-click to prevent weird states

    const targetNode = intersects[0].object.userData.node;
    const targetPosition = targetNode.sphere.position;

    // Calculate camera position to be 5 units away from the node
    const offset = new THREE.Vector3()
      .subVectors(camera.position, controls.target)
      .normalize()
      .multiplyScalar(5);
    const newCamPos = new THREE.Vector3().addVectors(targetPosition, offset);

    // Animate camera position
    new TWEEN.Tween(camera.position)
      .to(newCamPos, 1000) // 1 second animation
      .easing(TWEEN.Easing.Cubic.Out)
      .start();

    // Animate controls target
    new TWEEN.Tween(controls.target)
      .to(targetPosition, 1000)
      .easing(TWEEN.Easing.Cubic.Out)
      .start();
  }
}

renderer.domElement.addEventListener("mousemove", onMouseMove);
renderer.domElement.addEventListener("mousedown", onMouseDown);
renderer.domElement.addEventListener("mouseup", onMouseUp);
renderer.domElement.addEventListener("dblclick", onDoubleClick);
