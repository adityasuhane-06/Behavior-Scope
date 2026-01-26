import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Paper,
  LinearProgress,
  Alert,
  Card,
  CardContent
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function VideoUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const navigate = useNavigate();

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'];
      if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov)$/i)) {
        setError('Please select a valid video file (MP4, AVI, or MOV)');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      setProgress(0);

      // Create form data
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Upload and start analysis
      const response = await axios.post('http://localhost:8000/upload-analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { job_id } = response.data;
      setJobId(job_id);
      setUploading(false);
      setAnalyzing(true);

      // Poll for status
      pollAnalysisStatus(job_id);

    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
      setUploading(false);
    }
  };

  const pollAnalysisStatus = async (jobId) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/analysis-status/${jobId}`);
        const { status, progress: jobProgress, session_id, error: jobError } = response.data;

        setProgress(jobProgress || 0);

        if (status === 'completed') {
          clearInterval(interval);
          setAnalyzing(false);
          setSessionId(session_id);
          setProgress(100);
        } else if (status === 'failed') {
          clearInterval(interval);
          setAnalyzing(false);
          setError(jobError || 'Analysis failed');
        }
      } catch (err) {
        clearInterval(interval);
        setError('Failed to check analysis status');
        setAnalyzing(false);
      }
    }, 2000); // Poll every 2 seconds
  };

  const handleViewResults = () => {
    if (sessionId) {
      navigate(`/session/${sessionId}`);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, margin: '0 auto', mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Upload Video for Analysis
      </Typography>

      <Paper sx={{ p: 4, mt: 3 }}>
        {!sessionId && (
          <>
            <Box
              sx={{
                border: '2px dashed #ccc',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: 'action.hover'
                }
              }}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input
                id="file-input"
                type="file"
                accept="video/*,.mp4,.avi,.mov"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
              />
              <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6">
                {selectedFile ? selectedFile.name : 'Click to select a video file'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Supported formats: MP4, AVI, MOV
              </Typography>
            </Box>

            {selectedFile && !uploading && !analyzing && (
              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <Typography variant="body1" sx={{ mb: 2 }}>
                  File selected: <strong>{selectedFile.name}</strong>
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                  Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<CloudUploadIcon />}
                  onClick={handleUpload}
                >
                  Upload and Analyze
                </Button>
              </Box>
            )}
          </>
        )}

        {(uploading || analyzing) && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              {uploading ? 'Uploading...' : 'Analyzing...'}
            </Typography>
            <LinearProgress variant="determinate" value={progress} sx={{ mb: 1 }} />
            <Typography variant="body2" color="textSecondary">
              {progress}% complete
            </Typography>
            {analyzing && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Analysis is running. This may take several minutes depending on video length.
              </Alert>
            )}
          </Box>
        )}

        {sessionId && (
          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <CheckCircleIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              Analysis Complete!
            </Typography>
            <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
              Session ID: {sessionId}
            </Typography>
            <Button
              variant="contained"
              size="large"
              onClick={handleViewResults}
            >
              View Results
            </Button>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 3 }} icon={<ErrorIcon />}>
            {error}
          </Alert>
        )}
      </Paper>

      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            What happens during analysis?
          </Typography>
          <Typography variant="body2" component="div">
            <ol>
              <li>Audio extraction and voice activity detection</li>
              <li>Speaker diarization and speech transcription</li>
              <li>Video analysis (face detection, pose tracking)</li>
              <li>Eye contact and movement analysis</li>
              <li>Behavioral scoring and pattern detection</li>
              <li>Report generation and database storage</li>
            </ol>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}

export default VideoUpload;
