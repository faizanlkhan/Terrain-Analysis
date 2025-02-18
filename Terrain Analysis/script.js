async function loadPyodideAndPackages() {
    try {
        window.pyodide = await loadPyodide();
        await pyodide.loadPackage(["numpy", "matplotlib"]);

        console.log("✅ Pyodide & Matplotlib Loaded Successfully");
    } catch (error) {
        console.error("⚠️ Error loading Pyodide or Matplotlib:", error);
        alert("Failed to load Pyodide packages. Try refreshing.");
    }
}


// Call the function to initialize Pyodide
loadPyodideAndPackages();



async function processTIFF() {
    let fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select a TIFF file first!");
        return;
    }

    let file = fileInput.files[0];
    let reader = new FileReader();

    reader.onload = async function (event) {
        try {
            let arrayBuffer = event.target.result;
            let tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
            let image = await tiff.getImage();
            let width = image.getWidth();
            let height = image.getHeight();

            console.log(`Original Size: ${width} x ${height}`);

            // 🛑 Limit large files to prevent crashes
            let maxSize = 128; // Adjust based on performance
            let scale = Math.min(1, maxSize / Math.max(width, height));
            let newWidth = Math.round(width * scale);
            let newHeight = Math.round(height * scale);

            let rasterData = await image.readRasters({ samples: [0], interleave: true });
            let elevationData = new Float32Array(newWidth * newHeight);

            for (let i = 0; i < newHeight; i++) {
                for (let j = 0; j < newWidth; j++) {
                    let origX = Math.floor(j / scale);
                    let origY = Math.floor(i / scale);
                    elevationData[i * newWidth + j] = rasterData[origY * width + origX] || 0;
                }
            }

            console.log(`Resized to: ${newWidth} x ${newHeight}`);

            // ✅ Render 2D & 3D maps first
            plot2DMap(elevationData, newWidth, newHeight);
            create3DPlot(elevationData, newWidth, newHeight);

            // ✅ After successful 2D/3D, process erosion risk
            processTIFFforErosion(elevationData, newWidth, newHeight);

            processElevationData(elevationData, newWidth, newHeight);

            // **🔥 Auto-run Landform Classification after maps are plotted 🔥**
            classifyLandformsAndPlot(elevationData, newWidth, newHeight);
            
        } catch (error) {
            console.error("Error processing TIFF:", error);
            alert("Failed to process TIFF. Try a different file or reduce size.");
        }
    };

    reader.readAsArrayBuffer(file);
}

function plot2DMap(data, width, height) {
    console.log("Plotting 2D Map with Contours...");

    try {
        let x = Array.from({ length: width }, (_, i) => i);
        let y = Array.from({ length: height }, (_, i) => i);

        let elevationGrid = Array.from({ length: height }, (_, i) =>
            data.slice(i * width, (i + 1) * width)
        );

        let plotData = [
            {
                type: "heatmap",
                z: elevationGrid,
                x: x,
                y: y,
                colorscale: "Viridis",
                colorbar: { title: "Elevation (meters)" },
                hoverongaps: false,
                zsmooth: "best",
            },
            {
                type: "contour", // ✅ Adding contour layer
                z: elevationGrid,
                x: x,
                y: y,
                colorscale: "Black", // 🔥 Ensures contrast
                line: { width: 1 }, // Thin contour lines
                contours: {
                    start: Math.min(...data), // Min elevation
                    end: Math.max(...data),   // Max elevation
                    size: (Math.max(...data) - Math.min(...data)) / 15, // Auto step
                },
            },
        ];

        let layout = {
            title: "2D Elevation Map with Contours",
            xaxis: { title: "Longitude", scaleanchor: "y" },
            yaxis: { title: "Latitude" },
            autosize: true,
            width: 1200,
            height: 600,
            margin: { t: 40, l: 50, r: 50, b: 50 },
        };

        Plotly.newPlot("plotly2D", plotData, layout)
            .then(() => console.log("✅ 2D Map Rendered with Contours"))
            .catch(err => console.error("❌ Error rendering 2D map:", err));
    } catch (error) {
        console.error("❌ Unexpected error in plot2DMap:", error);
    }
}




function create3DPlot(data, width, height) {
    let x = Array.from({ length: width }, (_, i) => i);
    let y = Array.from({ length: height }, (_, i) => i);

    let plotData = [{
        type: "surface",
        x: x,
        y: y,
        z: Array.from({ length: height }, (_, i) => data.slice(i * width, (i + 1) * width)),
        colorscale: "Viridis"
    }];

    Plotly.newPlot("plotly3D", plotData, {
        title: "3D Terrain Map",
        scene: {
            xaxis_title: "Longitude",
            yaxis_title: "Latitude",
            zaxis_title: "Elevation (meters)"
        }
    }).catch(err => console.error("Error rendering 3D map:", err));
}

// 🔥 Optimized Erosion Risk Processing
async function processTIFFforErosion(elevationData, width, height) {
    try {
        let erosionRisk = computeErosionRisk(elevationData, width, height);
        plotErosionRisk(erosionRisk);
    } catch (error) {
        console.error("Error computing erosion risk:", error);
    }
}

function computeErosionRisk(elevationData, width, height) {
    let slope = new Float32Array(width * height);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let idx = y * width + x;
            let dx = (elevationData[idx + 1] - elevationData[idx - 1]) / 2;
            let dy = (elevationData[idx + width] - elevationData[idx - width]) / 2;
            slope[idx] = Math.sqrt(dx * dx + dy * dy);
        }
    }

    let sortedSlope = [...slope].sort((a, b) => a - b);
    let lowThreshold = sortedSlope[Math.floor(sortedSlope.length * 0.33)];
    let highThreshold = sortedSlope[Math.floor(sortedSlope.length * 0.66)];
    let riskMap = new Uint8Array(width * height);

    for (let i = 0; i < width * height; i++) {
        if (slope[i] < lowThreshold) riskMap[i] = 1; // Low risk
        else if (slope[i] < highThreshold) riskMap[i] = 2; // Moderate risk
        else riskMap[i] = 3; // High risk
    }

    return { riskMap, width, height };
}

function plotErosionRisk(riskData) {
    let { riskMap, width, height } = riskData;

    console.log("Plotting Erosion Risk Map with Contour...");

    let x = Array.from({ length: width }, (_, i) => i);
    let y = Array.from({ length: height }, (_, i) => i);

    // Convert 1D riskMap array into a 2D array
    let reshapedRiskMap = [];
    for (let i = 0; i < height; i++) {
        reshapedRiskMap.push(riskMap.slice(i * width, (i + 1) * width));
    }

    // 🟢 Heatmap for Erosion Risk
    let heatmap = {
        type: "heatmap",
        z: reshapedRiskMap,
        x: x,
        y: y,
        colorscale: [
            [0, "green"],  // Low risk
            [0.5, "orange"],  // Medium risk
            [1, "red"]  // High risk
        ],
        colorbar: {
            title: "Erosion Risk",
            tickvals: [1, 2, 3],
            ticktext: ["Low", "Moderate", "High"]
        },
        hoverongaps: false
    };

    // 🟠 Contour lines overlaying erosion risk
    let contour = {
        type: "contour",
        z: reshapedRiskMap,
        x: x,
        y: y,
        contours: {
            coloring: "lines", // 🟢 Only lines (no filled regions)
            showlabels: true,  // 🏷 Show elevation labels
            labelfont: { size: 12, color: "black" },
            start: 1, // Lowest risk level
            end: 3, // Highest risk level
            size: 1, // Step between contours
        },
        line: { color: "black", width: 1.5 }, // 🖤 Thicker black contour lines
        hoverinfo: "none"
    };

    let layout = {
        title: "Erosion Risk Map with Contour Lines",
        xaxis: { title: "Longitude", scaleanchor: "y" },
        yaxis: { title: "Latitude" },
        autosize: true,
        width: 600,
        height: 600,
        margin: { t: 50, l: 50, r: 50, b: 50 },
    };

    // 📌 Plot both heatmap & contour together
    Plotly.newPlot("erosionMap", [heatmap, contour], layout)
        .then(() => console.log("Erosion Map with Contour Rendered"))
        .catch(err => console.error("Error rendering Erosion Map:", err));
}


function computeElevationStats(elevationData, width, height) {
    let totalPixels = width * height;
    let sortedData = [...elevationData].sort((a, b) => a - b);
    
    let minElevation = sortedData[0];
    let maxElevation = sortedData[sortedData.length - 1];
    let avgElevation = sortedData.reduce((sum, val) => sum + val, 0) / totalPixels;
    let medianElevation = sortedData[Math.floor(totalPixels / 2)];
    let elevationRange = maxElevation - minElevation;
    
    let sumSquaredDiffs = sortedData.reduce((sum, val) => sum + Math.pow(val - avgElevation, 2), 0);
    let stdDeviation = Math.sqrt(sumSquaredDiffs / totalPixels);
    
    let slopeSum = 0, aspectSum = 0;
    let flatCount = 0, hillyCount = 0, mountainCount = 0;
    let contourCount = 0;

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let idx = y * width + x;
            let dx = (elevationData[idx + 1] - elevationData[idx - 1]) / 2;
            let dy = (elevationData[idx + width] - elevationData[idx - width]) / 2;
            let slope = Math.sqrt(dx * dx + dy * dy);
            
            if (slope < 5) flatCount++;
            else if (slope < 15) hillyCount++;
            else mountainCount++;

            let aspect = Math.atan2(dy, dx) * (180 / Math.PI);
            if (aspect < 0) aspect += 360;
            
            slopeSum += slope;
            aspectSum += aspect;
            
            if (slope > 10) contourCount++; // Rough estimate of contour density
        }
    }

    let flatPercentage = (flatCount / totalPixels) * 100;
    let hillyPercentage = (hillyCount / totalPixels) * 100;
    let mountainPercentage = (mountainCount / totalPixels) * 100;
    let avgSlope = slopeSum / totalPixels;
    let avgAspect = aspectSum / totalPixels;
    let contourDensity = contourCount / totalPixels;

    return {
        minElevation, maxElevation, avgElevation, medianElevation,
        stdDeviation, elevationRange, avgSlope, avgAspect,
        flatPercentage, hillyPercentage, mountainPercentage, contourDensity
    };
}

function displayElevationTable(stats) {
    let tableHTML = `
        <table border="1" style="width: 100%; text-align: center;">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Min Elevation (m)</td><td>${stats.minElevation.toFixed(2)}</td></tr>
            <tr><td>Max Elevation (m)</td><td>${stats.maxElevation.toFixed(2)}</td></tr>
            <tr><td>Avg Elevation (m)</td><td>${stats.avgElevation.toFixed(2)}</td></tr>
            <tr><td>Median Elevation (m)</td><td>${stats.medianElevation.toFixed(2)}</td></tr>
            <tr><td>Elevation Range (m)</td><td>${stats.elevationRange.toFixed(2)}</td></tr>
            <tr><td>Standard Deviation (m)</td><td>${stats.stdDeviation.toFixed(2)}</td></tr>
            <tr><td>Avg Slope (%)</td><td>${stats.avgSlope.toFixed(2)}</td></tr>
            <tr><td>Avg Aspect (°)</td><td>${stats.avgAspect.toFixed(2)}</td></tr>
            <tr><td>Flat Area (%)</td><td>${stats.flatPercentage.toFixed(2)}%</td></tr>
            <tr><td>Hilly Area (%)</td><td>${stats.hillyPercentage.toFixed(2)}%</td></tr>
            <tr><td>Mountainous Area (%)</td><td>${stats.mountainPercentage.toFixed(2)}%</td></tr>
            <tr><td>Contour Density</td><td>${stats.contourDensity.toFixed(4)}</td></tr>
        </table>`;

    document.getElementById("elevationTable").innerHTML = tableHTML;
}

function processElevationData(elevationData, width, height) {
    let stats = computeElevationStats(elevationData, width, height);
    displayElevationTable(stats);
}

// Ensure this function is called after processing the TIFF file
// processElevationData(elevationData, width, height);


// Function to classify landforms based on elevation
function classifyLandforms(elevationData, width, height) {
    const landformClasses = [];
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let elevation = elevationData[y][x];
            let label = '';
            
            if (elevation < 100) {
                label = 'Flat';
            } else if (elevation < 500) {
                label = 'Hilly';
            } else {
                label = 'Mountainous';
            }
            
            landformClasses.push({ x, y, elevation, label });
        }
    }
    
    return landformClasses;
}

// Function to add symbols/icons on the 2D elevation map
function classifyLandformsAndPlot() {
    if (!elevationData) {
        console.error("Elevation data not found!");
        return;
    }

    let rows = elevationData.length;
    let cols = elevationData[0].length;
    let landformSymbols = [];

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let elevation = elevationData[i][j];
            let symbol = '';

            if (elevation < 50) {
                symbol = '🏝️'; // Lowland
            } else if (elevation < 200) {
                symbol = '🌳'; // Hilly terrain
            } else {
                symbol = '⛰️'; // Mountainous region
            }

            landformSymbols.push({
                x: j, // Keep x as column index
                y: rows - i, // Flip y-axis to match image
                text: symbol
            });
        }
    }

    let traceHeatmap = {
        z: elevationData,
        type: 'heatmap',
        colorscale: 'Earth',
        name: 'Elevation'
    };

    let layout = {
        title: "Landform Classification",
        annotations: landformSymbols.map(d => ({
            x: d.x,
            y: d.y,
            text: d.text,
            showarrow: false,
            font: { size: 12 }
        })),
        xaxis: { title: "Longitude", range: [0, cols] },
        yaxis: { title: "Latitude", range: [0, rows] }
    };

    Plotly.react('landformMap', [traceHeatmap], layout);
}

// 🚀 Run automatically after elevation data is processed
setTimeout(classifyLandformsAndPlot, 1000);

