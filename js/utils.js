import * as THREE from "https://cdn.skypack.dev/three@0.132.2";

// Function to create a text sprite
export function createTextSprite(message, color = "#ffffff") {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  context.font = "Bold 60px Arial"; // Increased font size for better visibility
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";

  // Measure text width to set canvas size
  const metrics = context.measureText(message);
  const textWidth = metrics.width;
  const textHeight = 60; // Approximate font height

  canvas.width = textWidth + 20; // Add some padding
  canvas.height = textHeight + 20;

  // Redraw text on resized canvas
  context.font = "Bold 60px Arial";
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(message, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;

  const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
  const sprite = new THREE.Sprite(spriteMaterial);

  // Scale the sprite based on the canvas size to maintain aspect ratio
  sprite.scale.set(canvas.width * 0.01, canvas.height * 0.01, 1); // Adjust scale factor as needed

  return sprite;
}
