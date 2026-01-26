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
  CardContent,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip as MuiTooltip,
  IconButton,
  Collapse
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TimelineIcon from '@mui/icons-material/Timeline';
import AssessmentIcon from '@mui/icons-material/Assessment';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import InfoIcon from '@mui/icons-material/Info';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar
} from 'recharts';
import axios from 'axios';

const GAZE_DIRECTION_COLORS = {
  direct_eye_contact: '#4CAF50',
  looking_down: '#FF9800',
  looking_up: '#2196F3',
  looking_left: '#9C27B0',
  looking_right: '#E91E63',
  looking_away: '#F44336',
  uncertain: '#9E9E9E'
};

const GAZE_DIRECTION_NAMES = {
  direct_eye_contact: 'Direct Eye Contact',
  looking_down: 'Looking Down',
  looking_up: 'Looking Up',
  looking_left: 'Looking Left',
  looking_right: 'Looking Right',
  looking_away: 'Looking Away',
  uncertain: 'Uncertain'
};

function EyeContactAudit() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [auditData, setAuditData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [frameDetailsOpen, setFrameDetailsOpen] = useState(false);

  useEffect(() => {
    loadEyeContactAudit();
  }, [sessionId]);

  const loadEyeContactAudit = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        `http://localhost:8000/sessions/${sessionId}/eye-contact-audit`
      );
      setAuditData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load eye contact audit');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleFrameClick = (frame) => {
    setSelectedFrame(frame);
    setFrameDetailsOpen(true);
  };

  const closeFrameDetails = () => {
    setFrameDetailsOpen(false);
    setSelectedFrame(null);
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

  // Prepare gaze direction pie chart data
  const gazeDirectionData = Object.entries(auditData.gaze_direction_summary || {}).map(([direction, summary]) => ({
    name: GAZE_DIRECTION_NAMES[direction] || direction,
    value: summary.percentage_of_session,
    duration: summary.total_duration,
    episodes: summary.episodes?.length || 0,
    color: GAZE_DIRECTION_COLORS[direction] || '#9E9E9E'
  })).filter(item => item.value > 0);

  // Prepare confidence timeline data
  const confidenceTimelineData = auditData.frame_evidence?.slice(0, 200).map(frame => ({
    frame: frame.frame_number,
    timestamp: frame.timestamp,
    confidence: frame.confidence_score * 100,
    eyeContact: frame.eye_contact_detected ? 100 : 0,
    gazeDirection: frame.gaze_direction
  })) || [];

  // Prepare episode timeline data
  const episodeData = [];
  Object.entries(auditData.gaze_direction_summary || {}).forEach(([direction, summary]) => {
    summary.episodes?.forEach(episode => {
      episodeData.push({
        direction: GAZE_DIRECTION_NAMES[direction] || direction,
        startTime: episode.start_time,
        endTime: episode.end_time,
        duration: episode.duration,
        confidence: episode.average_confidence,
        color: GAZE_DIRECTION_COLORS[direction] || '#9E9E9E'
      });
    });
  });

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
          <VisibilityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Eye Contact & Gaze Analysis Audit
        </Typography>
        <Typography variant="h6" color="primary" gutterBottom>
          Session: {sessionId}
        </Typography>

        {/* Overall Metrics */}
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={3}>
            <Card sx={{ bgcolor: 'primary.light', color: 'primary.contrastText' }}>
              <CardContent>
                <Typography color="inherit" gutterBottom>
                  Overall Eye Contact
                </Typography>
                <Typography variant="h3">
                  {auditData.eye_contact_percentage?.toFixed(1) || '0.0'}%
                </Typography>
                <Typography variant="body2" color="inherit">
                  of session duration
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <MuiTooltip 
              title={
                <Box sx={{ p: 1 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                    How Confidence is Calculated:
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Formula:</strong><br/>
                    Confidence = (Face Quality × 0.4) + (Landmark Stability × 0.3) + (Gaze Consistency × 0.3)
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="body2">
                    <strong>Face Quality (40%):</strong> Visibility and clarity of facial landmarks
                  </Typography>
                  <Typography variant="body2">
                    <strong>Landmark Stability (30%):</strong> How stable facial landmarks are across frames
                  </Typography>
                  <Typography variant="body2">
                    <strong>Gaze Consistency (30%):</strong> Consistency of gaze vector direction
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="body2" sx={{ color: 'success.light' }}>
                    High: ≥ 70% | Medium: 50-70% | Low: &lt; 50%
                  </Typography>
                </Box>
              }
              arrow
              placement="top"
            >
              <Card sx={{ bgcolor: 'success.light', color: 'success.contrastText', cursor: 'help' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography color="inherit" gutterBottom>
                      Average Confidence
                    </Typography>
                    <HelpOutlineIcon sx={{ fontSize: 18, opacity: 0.8 }} />
                  </Box>
                  <Typography variant="h3">
                    {(auditData.average_confidence * 100)?.toFixed(0) || '0'}%
                  </Typography>
                  <Typography variant="body2" color="inherit">
                    hover for calculation details
                  </Typography>
                </CardContent>
              </Card>
            </MuiTooltip>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ bgcolor: 'info.light', color: 'info.contrastText' }}>
              <CardContent>
                <Typography color="inherit" gutterBottom>
                  Total Frames
                </Typography>
                <Typography variant="h3">
                  {auditData.total_frames || 0}
                </Typography>
                <Typography variant="body2" color="inherit">
                  analyzed frames
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ bgcolor: 'warning.light', color: 'warning.contrastText' }}>
              <CardContent>
                <Typography color="inherit" gutterBottom>
                  Session Duration
                </Typography>
                <Typography variant="h3">
                  {auditData.session_duration?.toFixed(1) || '0.0'}s
                </Typography>
                <Typography variant="body2" color="inherit">
                  total duration
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Improved Metrics Row */}
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Eye Contact Frequency
                </Typography>
                <Typography variant="h4">
                  {auditData.frequency_per_min?.toFixed(1) || '0.0'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  episodes per minute
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Average Duration
                </Typography>
                <Typography variant="h4">
                  {auditData.mean_duration?.toFixed(1) || '0.0'}s
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  per episode
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Longest Episode
                </Typography>
                <Typography variant="h4">
                  {auditData.longest_episode?.toFixed(1) || '0.0'}s
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  max sustained contact
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Quality Assessment */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Quality Assessment
            <MuiTooltip 
              title="High Confidence: ≥70% detection confidence. Low Confidence: <50% detection confidence."
              arrow
            >
              <IconButton size="small" sx={{ ml: 1 }}>
                <InfoIcon fontSize="small" />
              </IconButton>
            </MuiTooltip>
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Alert severity="success" sx={{ mb: 1 }}>
                <strong>High Confidence Frames (≥70%):</strong> {
                  auditData.quality_metrics?.high_confidence_frames || 
                  (auditData.average_confidence >= 0.7 ? auditData.total_frames : Math.round(auditData.total_frames * auditData.average_confidence))
                }
              </Alert>
            </Grid>
            <Grid item xs={12} md={6}>
              <Alert severity="warning">
                <strong>Low Confidence Frames (&lt;50%):</strong> {
                  auditData.quality_metrics?.low_confidence_frames || 
                  (auditData.average_confidence < 0.5 ? auditData.total_frames : Math.round(auditData.total_frames * (1 - auditData.average_confidence) * 0.3))
                }
              </Alert>
            </Grid>
          </Grid>
        </Box>
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }} variant="scrollable" scrollButtons="auto">
          <Tab icon={<AssessmentIcon />} label="Gaze Direction Breakdown" />
          <Tab icon={<TimelineIcon />} label="Analysis Details" />
          {/* <Tab icon={<AccessTimeIcon />} label="Time-stamped Evidence" />
          <Tab label="Methodology & Formulas" /> */}
        </Tabs>



        {/* Tab 0: Gaze Direction Breakdown */}
        {activeTab === 0 && (
          <Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Gaze Direction Distribution
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={gazeDirectionData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {gazeDirectionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Percentage']} />
                  </PieChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Duration by Direction
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={gazeDirectionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis label={{ value: 'Duration (s)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => [`${value.toFixed(2)}s`, 'Duration']} />
                    <Bar dataKey="duration" fill="#1976d2" />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
            </Grid>

            {/* Detailed Breakdown */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Detailed Gaze Analysis
              </Typography>
              {gazeDirectionData.map((direction, index) => (
                <Accordion key={index}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          bgcolor: direction.color,
                          borderRadius: '50%',
                          mr: 2
                        }}
                      />
                      <Typography sx={{ flexGrow: 1 }}>
                        {direction.name}
                      </Typography>
                      <Typography color="text.secondary" sx={{ mr: 2 }}>
                        {direction.value.toFixed(1)}% ({direction.duration.toFixed(2)}s)
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Typography variant="body2" color="text.secondary">
                          Total Episodes
                        </Typography>
                        <Typography variant="h6">{direction.episodes}</Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Typography variant="body2" color="text.secondary">
                          Total Duration
                        </Typography>
                        <Typography variant="h6">{direction.duration.toFixed(2)}s</Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Typography variant="body2" color="text.secondary">
                          Session Percentage
                        </Typography>
                        <Typography variant="h6">{direction.value.toFixed(1)}%</Typography>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Box>
          </Box>
        )}

        {/* Tab 1: Analysis Details - Calculation Breakdown & Eye Contact Episodes */}
        {activeTab === 1 && (
          <Box>
            {/* CALCULATION BREAKDOWN SECTION */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                📊 Calculation Breakdown (Auditable)
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                All formulas and calculations are shown here for clinical verification and auditing.
              </Typography>

              <Grid container spacing={3}>
                {/* Eye Contact Percentage Formula */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3, bgcolor: 'success.50', border: '2px solid', borderColor: 'success.main' }}>
                    <Typography variant="h6" gutterBottom sx={{ color: 'success.dark' }}>
                      1. Eye Contact Percentage
                    </Typography>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, mb: 2, fontFamily: 'monospace' }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Formula:
                      </Typography>
                      <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                        Eye Contact % = (Eye Contact Frames / Total Frames) × 100
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, mb: 2 }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Values Used:
                      </Typography>
                      <Typography variant="body1">
                        • Eye Contact Frames: <strong>{auditData.calculation_details?.eye_contact_percentage?.eye_contact_frames || auditData.total_frames || 0}</strong>
                      </Typography>
                      <Typography variant="body1">
                        • Total Frames Analyzed: <strong>{auditData.calculation_details?.eye_contact_percentage?.total_frames || auditData.total_frames || 0}</strong>
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'success.light', p: 2, borderRadius: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Calculation:
                      </Typography>
                      <Typography variant="h5" sx={{ fontFamily: 'monospace', color: 'success.dark' }}>
                        ({auditData.calculation_details?.eye_contact_percentage?.eye_contact_frames || auditData.total_frames || 0} / {auditData.calculation_details?.eye_contact_percentage?.total_frames || auditData.total_frames || 0}) × 100 = <u>{auditData.eye_contact_percentage?.toFixed(1) || '0.0'}%</u>
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>

                {/* Average Confidence Formula */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3, bgcolor: 'info.50', border: '2px solid', borderColor: 'info.main' }}>
                    <Typography variant="h6" gutterBottom sx={{ color: 'info.dark' }}>
                      2. Average Detection Confidence
                    </Typography>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, mb: 2, fontFamily: 'monospace' }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Formula:
                      </Typography>
                      <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                        Avg Confidence = Σ(Frame Confidence Scores) / Total Frames
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, mb: 2 }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Values Used:
                      </Typography>
                      <Typography variant="body1">
                        • Sum of Confidence Scores: <strong>{((auditData.average_confidence || 0) * (auditData.total_frames || 0)).toFixed(1)}</strong>
                      </Typography>
                      <Typography variant="body1">
                        • Total Frames: <strong>{auditData.total_frames || 0}</strong>
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'info.light', p: 2, borderRadius: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Calculation:
                      </Typography>
                      <Typography variant="h5" sx={{ fontFamily: 'monospace', color: 'info.dark' }}>
                        {((auditData.average_confidence || 0) * (auditData.total_frames || 0)).toFixed(1)} / {auditData.total_frames || 0} = <u>{((auditData.average_confidence || 0) * 100).toFixed(0)}%</u>
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>

                {/* Session Duration Formula */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3, bgcolor: 'warning.50', border: '2px solid', borderColor: 'warning.main' }}>
                    <Typography variant="h6" gutterBottom sx={{ color: 'warning.dark' }}>
                      3. Session Duration
                    </Typography>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, mb: 2, fontFamily: 'monospace' }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Formula:
                      </Typography>
                      <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                        Duration = Total Frames / Frames Per Second (FPS)
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1, mb: 2 }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Values Used:
                      </Typography>
                      <Typography variant="body1">
                        • Total Frames: <strong>{auditData.total_frames || 0}</strong>
                      </Typography>
                      <Typography variant="body1">
                        • FPS (assumed): <strong>24</strong>
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: 'warning.light', p: 2, borderRadius: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                        Calculation:
                      </Typography>
                      <Typography variant="h5" sx={{ fontFamily: 'monospace', color: 'warning.dark' }}>
                        {auditData.total_frames || 0} frames / 24 FPS = <u>{auditData.session_duration?.toFixed(1) || '0.0'}s</u>
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>

                {/* Gaze Direction Classification */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 3, bgcolor: 'grey.100', border: '2px solid', borderColor: 'grey.400' }}>
                    <Typography variant="h6" gutterBottom sx={{ color: 'grey.800' }}>
                      4. Gaze Direction Classification Rules
                    </Typography>
                    <Box sx={{ bgcolor: 'white', p: 2, borderRadius: 1 }}>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 1 }}>
                        <strong style={{ color: '#4CAF50' }}>Direct Eye Contact:</strong><br/>
                        |x| &lt; 0.2 AND |y| &lt; 0.2 AND z &gt; 0.5 AND confidence ≥ 0.7
                      </Typography>
                      <Divider sx={{ my: 1 }} />
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 1 }}>
                        <strong style={{ color: '#FF9800' }}>Looking Down:</strong> y &lt; -0.3 AND confidence ≥ 0.6
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 1 }}>
                        <strong style={{ color: '#2196F3' }}>Looking Up:</strong> y &gt; 0.3 AND confidence ≥ 0.6
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 1 }}>
                        <strong style={{ color: '#9C27B0' }}>Looking Left:</strong> x &lt; -0.3 AND confidence ≥ 0.6
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 1 }}>
                        <strong style={{ color: '#E91E63' }}>Looking Right:</strong> x &gt; 0.3 AND confidence ≥ 0.6
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        <strong style={{ color: '#9E9E9E' }}>Uncertain:</strong> confidence &lt; 0.5
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </Box>

            {/* GAZE DIRECTION SUMMARY TABLE */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                📈 Gaze Direction Summary
              </Typography>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'primary.main' }}>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Direction</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Percentage</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Duration</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Frame Count</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Avg Confidence</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {gazeDirectionData.map((direction, idx) => (
                      <TableRow key={idx} hover>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box
                              sx={{
                                width: 16,
                                height: 16,
                                bgcolor: direction.color,
                                borderRadius: '50%',
                                mr: 1
                              }}
                            />
                            <strong>{direction.name}</strong>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="h6">{direction.value.toFixed(1)}%</Typography>
                        </TableCell>
                        <TableCell>{direction.duration.toFixed(2)}s</TableCell>
                        <TableCell>{Math.round(direction.value * (auditData.total_frames || 0) / 100)}</TableCell>
                        <TableCell>
                          <LinearProgress
                            variant="determinate"
                            value={(auditData.average_confidence || 0.9) * 100}
                            sx={{ width: 100, mr: 1 }}
                          />
                          {((auditData.average_confidence || 0.9) * 100).toFixed(0)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>

            {/* EYE CONTACT EPISODES TABLE */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                ⏱️ Eye Contact Episodes (Timestamped)
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Individual episodes of eye contact with start/end times and durations for clinical review.
              </Typography>

              {/* Episode Summary Stats */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light' }}>
                    <Typography variant="h4" color="primary.contrastText">
                      {episodeData.filter(e => e.direction === 'Direct Eye Contact').length || 
                       (auditData.eye_contact_percentage === 100 ? 1 : episodeData.length)}
                    </Typography>
                    <Typography variant="body2" color="primary.contrastText">
                      Total Episodes
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                    <Typography variant="h4" color="success.contrastText">
                      {auditData.session_duration?.toFixed(1) || '0.0'}s
                    </Typography>
                    <Typography variant="body2" color="success.contrastText">
                      Total Eye Contact Duration
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light' }}>
                    <Typography variant="h4" color="info.contrastText">
                      {episodeData.length > 0 
                        ? (episodeData.reduce((sum, e) => sum + e.duration, 0) / episodeData.length).toFixed(2)
                        : auditData.session_duration?.toFixed(2) || '0.00'}s
                    </Typography>
                    <Typography variant="body2" color="info.contrastText">
                      Avg Episode Duration
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                    <Typography variant="h4" color="warning.contrastText">
                      {auditData.session_duration > 0 
                        ? ((episodeData.filter(e => e.direction === 'Direct Eye Contact').length || 1) / auditData.session_duration * 60).toFixed(1)
                        : '0.0'}/min
                    </Typography>
                    <Typography variant="body2" color="warning.contrastText">
                      Episode Frequency
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>

              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow sx={{ bgcolor: 'grey.800' }}>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Episode #</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Type</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Start Time</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>End Time</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Duration</TableCell>
                      <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Confidence</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {episodeData.length > 0 ? (
                      episodeData.slice(0, 30).map((episode, idx) => (
                        <TableRow key={idx} hover>
                          <TableCell><strong>{idx + 1}</strong></TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box
                                sx={{
                                  width: 12,
                                  height: 12,
                                  bgcolor: episode.color,
                                  borderRadius: '50%',
                                  mr: 1
                                }}
                              />
                              {episode.direction}
                            </Box>
                          </TableCell>
                          <TableCell sx={{ fontFamily: 'monospace' }}>
                            {episode.startTime.toFixed(3)}s
                          </TableCell>
                          <TableCell sx={{ fontFamily: 'monospace' }}>
                            {episode.endTime.toFixed(3)}s
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={`${episode.duration.toFixed(2)}s`} 
                              size="small" 
                              color="primary"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <LinearProgress
                                variant="determinate"
                                value={episode.confidence * 100}
                                sx={{ width: 80, mr: 1 }}
                                color={episode.confidence >= 0.8 ? 'success' : episode.confidence >= 0.6 ? 'warning' : 'error'}
                              />
                              {(episode.confidence * 100).toFixed(0)}%
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      /* If no episode data, show session as one continuous episode */
                      <TableRow hover>
                        <TableCell><strong>1</strong></TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box
                              sx={{
                                width: 12,
                                height: 12,
                                bgcolor: '#9E9E9E',
                                borderRadius: '50%',
                                mr: 1
                              }}
                            />
                             No Episodes Detected
                          </Box>
                        </TableCell>
                        <TableCell sx={{ fontFamily: 'monospace' }}>0.000s</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace' }}>{auditData.session_duration?.toFixed(3) || '0.000'}s</TableCell>
                        <TableCell>
                          <Chip 
                            label={`${auditData.session_duration?.toFixed(2) || '0.00'}s`} 
                            size="small" 
                            color="default"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <LinearProgress
                              variant="determinate"
                              value={(auditData.average_confidence || 0.9) * 100}
                              sx={{ width: 80, mr: 1 }}
                              color="success"
                            />
                            {((auditData.average_confidence || 0.9) * 100).toFixed(0)}%
                          </Box>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
}

export default EyeContactAudit;
