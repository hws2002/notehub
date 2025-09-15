Frontend Plan: Implementing the 3D Graph ViewThis document outlines the plan for creating the interactive 3D visualization for the knowledge graph. This plan directly follows the data processing steps laid out in Prototype_No_LLM.md.1. Goal & Technology ChoiceGoal: To render the extracted nodes and edges in an interactive, force-directed 3D space where the user can explore connections by rotating, panning, and zooming.Technology Choice: We will use the 3d-force-graph JavaScript library.Why? It is specifically designed for this exact purpose. It uses three.js under the hood but saves us a massive amount of development time by handling the 3D scene setup, physics simulation, and rendering logic automatically. This is ideal for rapid prototyping.Setup: We will load the library from a CDN, requiring no complex installation.2. HTML StructureThe foundation will be a single index.html file.<!DOCTYPE html>

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personalized Knowledge Graph</title>
    <style>
        body { margin: 0; }
        #graph-container {
            width: 100vw;
            height: 100vh;
        }
    </style>
</head>
<body>
    <!-- Container for the 3D graph -->
    <div id="graph-container"></div>

    <!-- Library Scripts -->
    <!-- We use the Kapsul library architecture which 3d-force-graph is built on -->
    <script src="//[unpkg.com/three](https://unpkg.com/three)"></script>
    <script src="//[unpkg.com/3d-force-graph](https://unpkg.com/3d-force-graph)"></script>

    <!-- Our Application Logic -->
    <script src="app.js"></script> <!-- We will place our logic in a separate file for clarity -->

</body>
</html>
Note: While the plan is to have logic in app.js, for the single-file prototype, we can place this logic inside <script> tags in the HTML file itself.3. Data TransformationThe 3d-force-graph library expects the data in a slightly different format than what we defined in the previous plan. We will need a simple transformation step.Our Nodes Array ({id: 1, label: 'React'}) is mostly compatible.Our Edges Array ({from: 1, to: 2}) needs to be transformed into {source: 1, target: 2}.<!-- end list -->// Example Transformation
const originalEdges = [{from: 1, to: 2}, {from: 1, to: 3}];

const transformedEdges = originalEdges.map(edge => ({
source: edge.from,
target: edge.to
})); 4. JavaScript Implementation StepsThis logic will be implemented after the node and edge extraction functions from Prototype_No_LLM.md have run.Get DOM Element: Get a reference to the <div id="graph-container">.Instantiate Graph: Create an instance of the 3d-force-graph.const graphContainer = document.getElementById('graph-container');
const myGraph = ForceGraph3D();
Configure and Load Data:Chain configuration methods to the instance to tell it how to render the data.Pass the transformed nodes and edges to the .graphData() method.<!-- end list -->// Assume `nodes` and `transformedEdges` are available from previous steps
myGraph(graphContainer)
.graphData({ nodes: nodes, links: transformedEdges })
.nodeLabel('label') // Show the 'label' property on hover
.nodeAutoColorBy('label'); // Automatically color nodes based on their label
Add Interactivity (Click Events):Implement a onNodeClick event handler. For the prototype, this can simply log the node's data. Later, this will trigger the sidebar to show details or call the LLM.<!-- end list -->myGraph.onNodeClick(node => {
// This is where we will add logic to display more info
console.log('Clicked on node:', node);

    // Center camera on the clicked node
    const distance = 40;
    const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
    myGraph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
        node, // lookAt ({ x, y, z })
        3000  // ms transition duration
    );

});
Adjust Physics/Forces (Optional):The library allows you to adjust the forces to control how the graph looks and behaves. We can add controls for this later.<!-- end list -->// Example: Make links longer and less rigid
myGraph.d3Force('link').distance(link => 30);
myGraph.d3Force('charge').strength(-120);
Add Interactivity & Persistent Labels (Advanced):To create persistent labels that are always visible, we must use .nodeThreeObject(). This overrides the default node rendering, so we need to create the sphere and the text label manually.The onNodeClick event handler can still be used to zoom in or perform other actions.myGraph.onNodeClick(node => {
// Center camera on the clicked node
const distance = 40;
const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
myGraph.cameraPosition(
{ x: node.x _ distRatio, y: node.y _ distRatio, z: node.z \* distRatio }, // new position
node, // lookAt ({ x, y, z })
3000 // ms transition duration
);
})
.nodeThreeObject(node => {
// Create a container group for our node and label
const group = new THREE.Group();

     // Create the node sphere
     const geometry = new THREE.SphereGeometry(2); // Adjust size as needed
     const material = new THREE.MeshLambertMaterial({
         color: node.color || 'blue',
         transparent: true,
         opacity: 0.8
     });
     const sphere = new THREE.Mesh(geometry, material);
     group.add(sphere);

     // Create the text label
     const text = node.label;
     if (text) {
         const canvas = document.createElement('canvas');
         const context = canvas.getContext('2d');
         context.font = 'Bold 10px Arial';
         const textWidth = context.measureText(text).width;
         canvas.width = textWidth;
         canvas.height = 12; // Font size + padding

         context.font = 'Bold 10px Arial';
         context.fillStyle = 'rgba(255, 255, 255, 0.95)';
         context.fillText(text, 0, 10);

         const texture = new THREE.CanvasTexture(canvas);
         const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
         const sprite = new THREE.Sprite(spriteMaterial);

         // Position the sprite to the side of the sphere
         sprite.scale.set(canvas.width/2, canvas.height/2, 1.0);
         sprite.position.set(5, 0, 0); // Adjust position relative to the sphere

         group.add(sprite);
     }

     return group;

});
Adjust Physics/Forces (Optional):
