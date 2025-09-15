import * as THREE from "https://cdn.skypack.dev/three@0.132.2";
import { createTextSprite } from "./utils.js";

// A class to represent a single node in our graph
export class GraphNode {
  constructor(id, name, position, color = 0xffffff, geometry, scene) {
    this.id = id;
    this.name = name;

    // --- Visuals ---
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.5,
      metalness: 0.5,
    });
    this.sphere = new THREE.Mesh(geometry, material);
    this.originalColor = new THREE.Color(color); // Store original color
    this.sphere.position.copy(position);
    scene.add(this.sphere);

    this.label = createTextSprite(name, "#ffffff");
    scene.add(this.label);

    this.sphere.userData.node = this; // Link back from the mesh to the node object

    // --- Physics Properties ---
    this.velocity = new THREE.Vector3(); // Start at rest
    this.force = new THREE.Vector3(); // Net force acting on the node
    this.connectedEdges = []; // To store connected edges
    this.previousPosition = new THREE.Vector3(); // For calculating velocity on release
  }

  // Method to keep the label in sync with the sphere
  update() {
    this.label.position.set(
      this.sphere.position.x,
      this.sphere.position.y + 1.2,
      this.sphere.position.z
    );
  }
}
