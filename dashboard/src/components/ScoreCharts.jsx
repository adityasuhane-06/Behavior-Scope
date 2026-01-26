import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { Box, Typography, Paper, Grid } from '@mui/material';

function ScoreCharts({ frames }) {
  // Process frames to extract detection data
  const detectionData = frames.map(frame => ({
    frame: frame.frame_number,
    timestamp: parseFloat(frame.timestamp || 0).toFixed(2),
    faceDetected: frame.face_detected ? 1 : 0,
    poseDetected: frame.pose_detected ? 1 : 0,
    eyeContactDetected: frame.eye_contact_detected ? 1 : 0,
    movementDetected: frame.movement_detected ? 1 : 0,
    confidenceScore: frame.confidence_score || 0
  }));

  // Calculate detection statistics
  const stats = {
    totalFrames: frames.length,
    faceDetections: frames.filter(f => f.face_detected).length,
    poseDetections: frames.filter(f => f.pose_detected).length,
    eyeContactDetections: frames.filter(f => f.eye_contact_detected).length,
    movementDetections: frames.filter(f => f.movement_detected).length
  };

  // Create summary data for bar chart
  const summaryData = [
    { name: 'Face', detections: stats.faceDetections, percentage: (stats.faceDetections / stats.totalFrames * 100).toFixed(1) },
    { name: 'Pose', detections: stats.poseDetections, percentage: (stats.poseDetections / stats.totalFrames * 100).toFixed(1) },
    { name: 'Eye Contact', detections: stats.eyeContactDetections, percentage: (stats.eyeContactDetections / stats.totalFrames * 100).toFixed(1) },
    { name: 'Movement', detections: stats.movementDetections, percentage: (stats.movementDetections / stats.totalFrames * 100).toFixed(1) }
  ];

  if (frames.length === 0) {
    return (
      <Box textAlign="center" py={4}>
        <Typography variant="h6" color="textSecondary">
          No frame data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Detection Summary */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Detection Summary
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={summaryData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'detections' ? `${value} frames` : `${value}%`,
                    name === 'detections' ? 'Detections' : 'Percentage'
                  ]}
                />
                <Legend />
                <Bar dataKey="detections" fill="#1976d2" name="Detections" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Statistics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Frame Analysis Statistics
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body1">
                <strong>Total Frames:</strong> {stats.totalFrames}
              </Typography>
              <Typography variant="body1">
                <strong>Face Detections:</strong> {stats.faceDetections} ({(stats.faceDetections / stats.totalFrames * 100).toFixed(1)}%)
              </Typography>
              <Typography variant="body1">
                <strong>Pose Detections:</strong> {stats.poseDetections} ({(stats.poseDetections / stats.totalFrames * 100).toFixed(1)}%)
              </Typography>
              <Typography variant="body1">
                <strong>Eye Contact Events:</strong> {stats.eyeContactDetections} ({(stats.eyeContactDetections / stats.totalFrames * 100).toFixed(1)}%)
              </Typography>
              <Typography variant="body1">
                <strong>Movement Events:</strong> {stats.movementDetections} ({(stats.movementDetections / stats.totalFrames * 100).toFixed(1)}%)
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Detection Timeline */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Detection Timeline
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={detectionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="frame"
                  label={{ value: 'Frame Number', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  label={{ value: 'Detection (0/1)', angle: -90, position: 'insideLeft' }}
                  domain={[0, 1]}
                />
                <Tooltip 
                  formatter={(value, name) => [
                    value ? 'Detected' : 'Not Detected',
                    name.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
                  ]}
                />
                <Legend />
                <Line
                  type="stepAfter"
                  dataKey="faceDetected"
                  stroke="#1976d2"
                  name="Face"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="poseDetected"
                  stroke="#388e3c"
                  name="Pose"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="eyeContactDetected"
                  stroke="#f57c00"
                  name="Eye Contact"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="movementDetected"
                  stroke="#d32f2f"
                  name="Movement"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Confidence Scores */}
        {detectionData.some(d => d.confidenceScore > 0) && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Confidence Scores Over Time
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={detectionData.filter(d => d.confidenceScore > 0)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frame" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="confidenceScore"
                    stroke="#9c27b0"
                    name="Confidence"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default ScoreCharts;
