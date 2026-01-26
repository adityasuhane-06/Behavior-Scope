import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  CircularProgress,
  Card,
  CardContent
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { fetchStatistics } from '../api/auditAPI';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function Statistics() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      const data = await fetchStatistics();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch statistics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!stats) {
    return (
      <Box textAlign="center" mt={4}>
        <Typography variant="h6" color="error">
          Failed to load statistics
        </Typography>
      </Box>
    );
  }

  // Prepare data for charts
  const detectionData = [
    { name: 'Face', count: stats.total_face_detections || 0 },
    { name: 'Pose', count: stats.total_pose_detections || 0 },
    { name: 'Eye Contact', count: stats.total_eye_contact_detections || 0 },
    { name: 'Movement', count: stats.total_movement_detections || 0 }
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Statistics
      </Typography>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Sessions
              </Typography>
              <Typography variant="h4">
                {stats.total_sessions || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Frames Analyzed
              </Typography>
              <Typography variant="h4">
                {stats.total_frames || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg. Engagement Score
              </Typography>
              <Typography variant="h4">
                {stats.avg_engagement_score
                  ? `${(stats.avg_engagement_score * 100).toFixed(1)}%`
                  : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg. Autism Risk Score
              </Typography>
              <Typography variant="h4">
                {stats.avg_autism_risk_score
                  ? `${(stats.avg_autism_risk_score * 100).toFixed(1)}%`
                  : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Detection Counts by Type
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={detectionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Detection Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={detectionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {detectionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Average Scores
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Box textAlign="center" p={2}>
                  <Typography variant="h5" color="primary">
                    {stats.avg_engagement_score
                      ? `${(stats.avg_engagement_score * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Engagement Score
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box textAlign="center" p={2}>
                  <Typography variant="h5" color="error">
                    {stats.avg_autism_risk_score
                      ? `${(stats.avg_autism_risk_score * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Autism Risk Score
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box textAlign="center" p={2}>
                  <Typography variant="h5" color="success.main">
                    {stats.avg_eye_contact_score
                      ? `${(stats.avg_eye_contact_score * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Eye Contact Score
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box textAlign="center" p={2}>
                  <Typography variant="h5" color="warning.main">
                    {stats.avg_movement_score
                      ? `${(stats.avg_movement_score * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Movement Score
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Statistics;
