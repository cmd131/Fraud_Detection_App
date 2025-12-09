# Setup Instructions

This guide provides step-by-step instructions for setting up and running the Fraud/Spam Text Detection project.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd fraud-spam-detection

# Or download and extract the ZIP file, then navigate to the project directory
```

### Step 2: Set Up the Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

5. Start the backend server:
```bash
python app.py
```

The backend API should now be running at `http://127.0.0.1:5000`

### Step 3: Set Up the Frontend

1. Open a new terminal window/tab (keep the backend running)

2. Navigate to the frontend directory:
```bash
cd frontend
```

3. Serve the frontend application:

**Option A: Using Python's built-in server**
```bash
python -m http.server 3000
```

**Option B: Using Node.js http-server (if installed)**
```bash
npx http-server -p 3000
```

**Option C: Open directly in browser**
- Simply open `index.html` in your web browser

4. Access the web interface at `http://127.0.0.1:3000/`

### Step 4: Verify Installation

1. Ensure the backend is running and accessible at `http://127.0.0.1:5000`

2. Test the API using the provided test script:
```bash
cd backend
python test_api.py
```

3. Open the frontend in your browser and try submitting a test message

## Next Steps

After successful installation:
1. Review the README.md for usage instructions
2. Watch the demo video to see the system in action


