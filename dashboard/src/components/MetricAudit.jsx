import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import axios from 'axios';
import EyeContactAudit from './EyeContactAudit';
import EyeContactAuditTest from './EyeContactAuditTest';
import EyeContactAuditSimple from './EyeContactAuditSimple';
import StutteringAudit from './StutteringAudit';
import TurnTakingAudit from './TurnTakingAudit';
import ResponsivenessAudit from './ResponsivenessAudit';

function MetricAudit() {
  const { sessionId, metricName } = useParams();
  const navigate = useNavigate();
  const [auditData, setAuditData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Route to specialized audit components based on metric name
  if (metricName === 'eye_contact') {
    return <EyeContactAudit />;
  }
  if (metricName === 'stuttering') {
    return <StutteringAudit />;
  }
  if (metricName === 'turn_taking') {
    return <TurnTakingAudit />;
  }
  if (metricName === 'responsiveness') {
    return <ResponsivenessAudit />;
  }

  useEffect(() => {
    loadMetricAudit();
  }, [sessionId, metricName]);

  const loadMetricAudit = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        `http://localhost:8000/sessions/${sessionId}/metric/${metricName}`
      );
      setAuditData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load metric audit');
    } finally {
      setLoading(false);
    }
  };

  const getMetricDisplayName = (name) => {
    const names = {
      vocal_regulation: 'Vocal Regulation Index (VRI)',
      motor_agitation: 'Motor Agitation Index (MAI)',
      eye_contact: 'Eye Contact Score',
      attention_stability: 'Attention Stability Score (ASS)',
      social_engagement: 'Social Engagement Index (SEI)',
      facial_affect: 'Facial Affect Index (FAI)'
    };
    return names[name] || name;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box textAlign="center" mt={4}>
        <Typography variant="h6" color="error">
          {error}
        </Typography>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(`/session/${sessionId}`)}
          sx={{ mt: 2 }}
        >
          Back to Session
        </Button>
      </Box>
    );
  }

  if (!auditData) {
    return null;
  }

  // Prepare chart data
  const chartData = auditData.frame_data
    .filter(frame => frame.behavioral_scores)
    .slice(0, 100) // Limit to first 100 frames for performance
    .map(frame => ({
      frame: frame.frame_number,
      timestamp: frame.timestamp.toFixed(2),
      score: (frame.behavioral_scores[metricName + '_score'] || 0) * 100
    }));

  return (
    <Box>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(`/session/${sessionId}`)}
        sx={{ mb: 2 }}
      >
        Back to Session
      </Button>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          {getMetricDisplayName(metricName)}
        </Typography>
        <Typography variant="h6" color="primary" gutterBottom>
          Session: {sessionId}
        </Typography>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Overall Score
                </Typography>
                <Typography variant="h3">
                  {auditData.overall_score !== null
                    ? auditData.overall_score.toFixed(1)
                    : 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Total Frames Analyzed
                </Typography>
                <Typography variant="h3">
                  {auditData.frame_data.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Analysis Operations
                </Typography>
                <Typography variant="h3">
                  {auditData.operations.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {chartData.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Score Over Time
          </Typography>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="frame"
                label={{ value: 'Frame Number', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }}
                domain={[0, 100]}
              />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#1976d2"
                name={getMetricDisplayName(metricName)}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Operation Log
        </Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell><strong>Timestamp</strong></TableCell>
                <TableCell><strong>Stage</strong></TableCell>
                <TableCell><strong>Operation</strong></TableCell>
                <TableCell><strong>Status</strong></TableCell>
                <TableCell><strong>Details</strong></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {auditData.operations.map((op, idx) => (
                <TableRow key={idx} hover>
                  <TableCell>{new Date(op.timestamp).toLocaleString()}</TableCell>
                  <TableCell>
                    <Chip label={op.stage} size="small" />
                  </TableCell>
                  <TableCell>{op.operation}</TableCell>
                  <TableCell>
                    <Chip
                      label={op.status}
                      size="small"
                      color={op.status === 'success' ? 'success' : 'error'}
                    />
                  </TableCell>
                  <TableCell>{op.details}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
}

export default MetricAudit;
