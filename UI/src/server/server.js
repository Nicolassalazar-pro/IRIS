const express = require('express');
const multer = require('multer');
const https = require('https');
const http = require('http'); // Also create an HTTP server for redirect
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 6969;
const httpPort = 6970; // Port for HTTP redirects

const httpsOptions = {
  key: fs.readFileSync(path.join(__dirname, 'certs', 'localhost+2-key.pem')),
  cert: fs.readFileSync(path.join(__dirname, 'certs', 'localhost+2.pem'))
};

// Startup cleanup function
const startupCleanup = () => {
  const uploadsDir = path.join(__dirname, 'uploads');
  
  // Create uploads directory if it doesn't exist
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
    console.log('Created uploads directory');
    return;
  }
  
  // Check for and delete any existing audio file
  const audioPath = path.join(uploadsDir, 'audio.wav');
  if (fs.existsSync(audioPath)) {
    try {
      fs.unlinkSync(audioPath);
      console.log('Cleaned up existing audio file on startup');
    } catch (err) {
      console.error('Error cleaning up file:', err);
    }
  }
};

// Run cleanup at startup
startupCleanup();

// Configure CORS to allow all origins with credentials
app.use(cors({
  origin: function(origin, callback) {
    // Allow any origin
    callback(null, true);
  },
  credentials: true
}));

app.use(express.json());

// Configure storage for uploaded files
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function(req, file, cb) {
    cb(null, 'audio.wav');
  }
});

const upload = multer({ storage: storage });

// Serve uploaded files with appropriate headers
app.use('/uploads', express.static(path.join(__dirname, 'uploads'), {
  // Set headers to prevent caching
  setHeaders: function (res, path) {
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate, private');
    res.set('Pragma', 'no-cache');
    res.set('Expires', '0');
    // Add CORS headers specifically for file serving
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  }
}));

// Endpoint to upload audio file
app.post('/upload-audio', upload.single('audio'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded');
  }
  
  res.json({
    success: true,
    message: 'File uploaded successfully',
    audioUrl: `https://10.53.1.209:${port}/uploads/audio.wav`
  });
});

// Endpoint to delete the current audio file after playback
app.post('/delete-audio', (req, res) => {
  const audioPath = path.join(__dirname, 'uploads', 'audio.wav');
  
  fs.access(audioPath, fs.constants.F_OK, (err) => {
    if (err) {
      // File doesn't exist
      return res.status(404).json({
        success: false,
        message: 'No audio file found to delete'
      });
    }
    
    // File exists, try to delete it
    fs.unlink(audioPath, (err) => {
      if (err) {
        return res.status(500).json({
          success: false,
          message: 'Error deleting audio file',
          error: err.message
        });
      }
      
      res.json({
        success: true,
        message: 'Audio file deleted successfully'
      });
    });
  });
});

// Endpoint to check and clean up any existing audio files
app.get('/cleanup', (req, res) => {
  const audioPath = path.join(__dirname, 'uploads', 'audio.wav');
  
  fs.access(audioPath, fs.constants.F_OK, (err) => {
    if (!err) {
      // File exists, delete it
      fs.unlink(audioPath, (unlinkErr) => {
        if (unlinkErr) {
          return res.status(500).json({ 
            success: false, 
            message: 'Error cleaning up file', 
            error: unlinkErr.message 
          });
        }
        res.json({ success: true, message: 'Cleaned up existing audio file' });
      });
    } else {
      res.json({ success: true, message: 'No existing audio file found' });
    }
  });
});

// Status endpoint
app.get('/status', (req, res) => {
  res.json({ status: 'Server is running with HTTPS' });
});

// Create HTTP server to redirect to HTTPS
const httpApp = express();
httpApp.get('*', (req, res) => {
  res.redirect(`https://10.53.1.209:${port}${req.url}`);
});

// Start both servers
https.createServer(httpsOptions, app).listen(port, '0.0.0.0', () => {
  console.log(`HTTPS Server running at https://10.53.1.209:${port}`);
});

http.createServer(httpApp).listen(httpPort, '0.0.0.0', () => {
  console.log(`HTTP Server running at https://10.53.1.209:${httpPort} (redirects to HTTPS)`);
});