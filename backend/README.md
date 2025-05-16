# Military Performance Prediction API

A FastAPI backend service for predicting military efficiency and strike winning rates using machine learning models.

## Features

- Real-time predictions via WebSockets
- REST API endpoints for one-time predictions
- Pre-trained models for efficiency and strike efficiency prediction
- Sample data generation for testing

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install fastapi uvicorn pandas numpy tensorflow scikit-learn joblib websockets
```

3. Run the server:

```bash
python app.py
```

The server will start on `http://localhost:8000`.

## WebSocket Testing Guide

### Using Browser-based WebSocket Client

1. Install a WebSocket client extension for your browser:
   - For Chrome: [Simple WebSocket Client](https://chrome.google.com/webstore/detail/simple-websocket-client/pfdhoblngboilpfeibdedpjgfnlcodoo)
   - For Firefox: [Websocket Client](https://addons.mozilla.org/en-US/firefox/addon/websocket-client/)

2. Connect to the WebSocket endpoints:
   - Efficiency prediction: `ws://localhost:8000/ws/predict/efficiency`
   - Strike efficiency prediction: `ws://localhost:8000/ws/predict/strike-efficiency`

3. Send test messages in JSON format (examples below)

### Using Python WebSocket Client

Create a new file called `test_websocket.py`:

```python
import asyncio
import websockets
import json

async def test_efficiency_prediction():
    # Connect to the WebSocket
    async with websockets.connect("ws://localhost:8000/ws/predict/efficiency") as websocket:
        # Sample soldier data
        test_data = {
            "Age": 28,
            "Experience": 5,
            "Training": 350,
            "Physical_Fitness": 0.75,
            "Mental_Readiness": 0.85,
            "Equipment_Quality": 0.8,
            "Strength": 82,
            "Speed": 78,
            "Endurance": 85,
            "TeamCoordination": 0.7
        }
        
        # Send the data as JSON
        await websocket.send(json.dumps(test_data))
        
        # Wait for a response
        response = await websocket.recv()
        
        # Parse and print the response
        result = json.loads(response)
        print(f"Efficiency prediction: {result}")

async def test_strike_efficiency_prediction():
    # Connect to the WebSocket
    async with websockets.connect("ws://localhost:8000/ws/predict/strike-efficiency") as websocket:
        # Sample soldier data
        test_data = {
            "Age": 30,
            "Weight": 75,
            "Height": 180,
            "Strength": 85,
            "Speed": 80,
            "Endurance": 75,
            "TrainingHours": 2000,
            "CombatExperience": 7,
            "Fatigue": 20,
            "Stress": 30,
            "WeaponAccuracy": 85,
            "WeaponProficiency": 80,
            "WeaponType": "Assault Rifle",
            "Terrain": "Urban",
            "TimeOfDay": "Day",
            "Weather": "Clear"
        }
        
        # Send the data as JSON
        await websocket.send(json.dumps(test_data))
        
        # Wait for a response
        response = await websocket.recv()
        
        # Parse and print the response
        result = json.loads(response)
        print(f"Strike efficiency prediction: {result}")

# Run both tests
async def main():
    print("Testing Efficiency Prediction...")
    await test_efficiency_prediction()
    
    print("\nTesting Strike Efficiency Prediction...")
    await test_strike_efficiency_prediction()

if __name__ == "__main__":
    asyncio.run(main())
```

Run the test script:

```bash
python test_websocket.py
```

### Using Web-based UI for Testing

You can create a simple HTML page to test the WebSocket connections:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Military Performance WebSocket Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { margin-bottom: 30px; }
        pre { background-color: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
        button { padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        input, select { padding: 8px; margin: 5px 0; width: 100%; }
        .result { margin-top: 15px; }
    </style>
</head>
<body>
    <h1>Military Performance WebSocket Test</h1>
    
    <div class="container">
        <h2>Efficiency Prediction</h2>
        <select id="efficiencyEndpoint">
            <option value="ws://localhost:8000/ws/predict/efficiency">WebSocket</option>
            <option value="http://localhost:8000/predict/efficiency">REST API</option>
        </select>
        <div>
            <textarea id="efficiencyData" rows="12" cols="60">{
  "Age": 28,
  "Experience": 5,
  "Training": 350,
  "Physical_Fitness": 0.75,
  "Mental_Readiness": 0.85,
  "Equipment_Quality": 0.8,
  "Strength": 82,
  "Speed": 78,
  "Endurance": 85,
  "TeamCoordination": 0.7
}</textarea>
        </div>
        <button id="sendEfficiency">Send Request</button>
        <div class="result">
            <h3>Result:</h3>
            <pre id="efficiencyResult">Results will appear here...</pre>
        </div>
    </div>
    
    <div class="container">
        <h2>Strike Efficiency Prediction</h2>
        <select id="strikeEndpoint">
            <option value="ws://localhost:8000/ws/predict/strike-efficiency">WebSocket</option>
            <option value="http://localhost:8000/predict/strike-efficiency">REST API</option>
        </select>
        <div>
            <textarea id="strikeData" rows="16" cols="60">{
  "Age": 30,
  "Weight": 75,
  "Height": 180,
  "Strength": 85,
  "Speed": 80,
  "Endurance": 75,
  "TrainingHours": 2000,
  "CombatExperience": 7,
  "Fatigue": 20,
  "Stress": 30,
  "WeaponAccuracy": 85,
  "WeaponProficiency": 80,
  "WeaponType": "Assault Rifle",
  "Terrain": "Urban",
  "TimeOfDay": "Day",
  "Weather": "Clear"
}</textarea>
        </div>
        <button id="sendStrike">Send Request</button>
        <div class="result">
            <h3>Result:</h3>
            <pre id="strikeResult">Results will appear here...</pre>
        </div>
    </div>

    <script>
        // WebSocket connections
        let efficiencyWs = null;
        let strikeWs = null;

        // Handle efficiency prediction
        document.getElementById('sendEfficiency').addEventListener('click', function() {
            const endpointUrl = document.getElementById('efficiencyEndpoint').value;
            const data = document.getElementById('efficiencyData').value;
            
            try {
                const jsonData = JSON.parse(data);
                
                if (endpointUrl.startsWith('ws://')) {
                    // WebSocket request
                    if (efficiencyWs) {
                        efficiencyWs.close();
                    }
                    
                    efficiencyWs = new WebSocket(endpointUrl);
                    
                    efficiencyWs.onopen = function() {
                        efficiencyWs.send(data);
                    };
                    
                    efficiencyWs.onmessage = function(event) {
                        document.getElementById('efficiencyResult').textContent = JSON.stringify(JSON.parse(event.data), null, 2);
                    };
                    
                    efficiencyWs.onerror = function(error) {
                        document.getElementById('efficiencyResult').textContent = 'WebSocket Error: ' + JSON.stringify(error);
                    };
                } else {
                    // REST API request
                    fetch(endpointUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: data
                    })
                    .then(response => response.json())
                    .then(result => {
                        document.getElementById('efficiencyResult').textContent = JSON.stringify(result, null, 2);
                    })
                    .catch(error => {
                        document.getElementById('efficiencyResult').textContent = 'Error: ' + error;
                    });
                }
            } catch (error) {
                document.getElementById('efficiencyResult').textContent = 'Invalid JSON: ' + error;
            }
        });

        // Handle strike efficiency prediction
        document.getElementById('sendStrike').addEventListener('click', function() {
            const endpointUrl = document.getElementById('strikeEndpoint').value;
            const data = document.getElementById('strikeData').value;
            
            try {
                const jsonData = JSON.parse(data);
                
                if (endpointUrl.startsWith('ws://')) {
                    // WebSocket request
                    if (strikeWs) {
                        strikeWs.close();
                    }
                    
                    strikeWs = new WebSocket(endpointUrl);
                    
                    strikeWs.onopen = function() {
                        strikeWs.send(data);
                    };
                    
                    strikeWs.onmessage = function(event) {
                        document.getElementById('strikeResult').textContent = JSON.stringify(JSON.parse(event.data), null, 2);
                    };
                    
                    strikeWs.onerror = function(error) {
                        document.getElementById('strikeResult').textContent = 'WebSocket Error: ' + JSON.stringify(error);
                    };
                } else {
                    // REST API request
                    fetch(endpointUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: data
                    })
                    .then(response => response.json())
                    .then(result => {
                        document.getElementById('strikeResult').textContent = JSON.stringify(result, null, 2);
                    })
                    .catch(error => {
                        document.getElementById('strikeResult').textContent = 'Error: ' + error;
                    });
                }
            } catch (error) {
                document.getElementById('strikeResult').textContent = 'Invalid JSON: ' + error;
            }
        });
    </script>
</body>
</html>
```

Save this as `websocket_test.html` and open it in your browser.

## API Endpoints

### WebSocket Endpoints

- `/ws/predict/efficiency` - Predict soldier efficiency
- `/ws/predict/strike-efficiency` - Predict soldier strike winning rate

### REST API Endpoints

- `POST /predict/efficiency` - Predict soldier efficiency
- `POST /predict/strike-efficiency` - Predict soldier strike winning rate
- `GET /generate-data?data_type=efficiency&num_samples=10` - Generate sample efficiency data
- `GET /generate-data?data_type=strike&num_samples=10` - Generate sample strike data
- `GET /health` - Check API health status

## Sample Data Format

### Efficiency Prediction Data

```json
{
  "Age": 28,
  "Experience": 5,
  "Training": 350,
  "Physical_Fitness": 0.75,
  "Mental_Readiness": 0.85,
  "Equipment_Quality": 0.8,
  "Strength": 82,
  "Speed": 78,
  "Endurance": 85,
  "TeamCoordination": 0.7
}
```

### Strike Efficiency Prediction Data

```json
{
  "Age": 30,
  "Weight": 75,
  "Height": 180,
  "Strength": 85,
  "Speed": 80,
  "Endurance": 75,
  "TrainingHours": 2000,
  "CombatExperience": 7,
  "Fatigue": 20,
  "Stress": 30,
  "WeaponAccuracy": 85,
  "WeaponProficiency": 80,
  "WeaponType": "Assault Rifle",
  "Terrain": "Urban",
  "TimeOfDay": "Day",
  "Weather": "Clear"
}
```

## Troubleshooting

1. **Connection refused**: Make sure the FastAPI server is running
2. **Error in prediction**: Check the model files exist in the correct directories
3. **Models not loading**: Verify TensorFlow is installed correctly
4. **Invalid data format**: Ensure your JSON data matches the expected format

## Notes

- The models will fall back to random predictions if the pre-trained models are not found
- You can generate test data using the `/generate-data` endpoint 