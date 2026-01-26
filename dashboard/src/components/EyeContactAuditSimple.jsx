import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Button,
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import VisibilityIcon from '@mui/icons-material/Visibility';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
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

function getThresholdPurpose(key) {
  const purposes = {
    'eye_contact_confidence_threshold': 'Minimum confidence required to classify as direct eye contact',
    'gaze_angle_threshold_degrees': 'Maximum angle deviation from camera for eye contact detection',
    'face_quality_threshold': 'Minimum face landmark quality for reliable gaze estimation',
    'looking_down_y_threshold': 'Y-axis threshold for downward gaze classification',
    'looking_up_y_threshold': 'Y-axis threshold for upward gaze classification',
    'looking_left_x_threshold': 'X-axis threshold for leftward gaze classification',
    'looking_right_x_threshold': 'X-axis threshold for rightward gaze classification'
  };
  return purposes[key] || 'Detection parameter for gaze classification';
}

function EyeContactAuditSimple() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [auditData, setAuditData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);

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

  // Extract metrics from either root level or enhanced_metrics
  const eyeContactPercentage = auditData.enhanced_metrics?.eye_contact_percentage 
    || auditData.eye_contact_percentage 
    || 0;
  
  const averageConfidence = auditData.enhanced_metrics?.average_confidence 
    || auditData.average_confidence 
    || 0;
  
  const totalFrames = auditData.enhanced_metrics?.total_frames 
    || auditData.total_frames 
    || 0;
  
  const sessionDuration = auditData.session_duration 
    || (totalFrames / 24.0) 
    || 0;

  // Prepare gaze direction pie chart data
  const gazeDirectionData = Object.entries(auditData.gaze_direction_summary || {}).map(([direction, summary]) => ({
    name: GAZE_DIRECTION_NAMES[direction] || direction,
    value: summary.percentage_of_session || 0,
    color: GAZE_DIRECTION_COLORS[direction] || '#9E9E9E'
  })).filter(item => item.value > 0);

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
                  {eyeContactPercentage?.toFixed(1) || '0.0'}%
                </Typography>
                <Typography variant="body2" color="inherit">
                  of session duration
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ bgcolor: 'success.light', color: 'success.contrastText' }}>
              <CardContent>
                <Typography color="inherit" gutterBottom>
                  Average Confidence
                </Typography>
                <Typography variant="h3">
                  {((averageConfidence || 0) * 100)?.toFixed(0)}%
                </Typography>
                <Typography variant="body2" color="inherit">
                  detection confidence
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card sx={{ bgcolor: 'info.light', color: 'info.contrastText' }}>
              <CardContent>
                <Typography color="inherit" gutterBottom>
                  Total Frames
                </Typography>
                <Typography variant="h3">
                  {totalFrames || 0}
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
                  {sessionDuration?.toFixed(1) || '0.0'}s
                </Typography>
                <Typography variant="body2" color="inherit">
                  total duration
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
          <Tab label="Gaze Direction Breakdown" />
          <Tab label="Analysis Details" />
          <Tab label="Methodology & Formulas" />
        </Tabs>

        {/* Tab 0: Gaze Direction Breakdown */}
        {activeTab === 0 && (
          <Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Gaze Direction Distribution
                </Typography>
                {gazeDirectionData.length > 0 ? (
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
                ) : (
                  <Typography variant="body1" color="text.secondary">
                    No gaze direction data available
                  </Typography>
                )}
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Gaze Direction Summary
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Direction</strong></TableCell>
                        <TableCell><strong>Percentage</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {gazeDirectionData.map((direction, idx) => (
                        <TableRow key={idx}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box
                                sx={{
                                  width: 12,
                                  height: 12,
                                  bgcolor: direction.color,
                                  borderRadius: '50%',
                                  mr: 1
                                }}
                              />
                              {direction.name}
                            </Box>
                          </TableCell>
                          <TableCell>{direction.value.toFixed(1)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Tab 1: Analysis Details */}
        {activeTab === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Analysis Details
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="text.secondary">
                  Detection Method
                </Typography>
                <Typography variant="body1">
                  {auditData.detection_method || 'Unknown'}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="text.secondary">
                  Enhanced Data Available
                </Typography>
                <Chip
                  label={auditData.enhanced_data_available ? 'Yes' : 'No'}
                  color={auditData.enhanced_data_available ? 'success' : 'warning'}
                />
              </Grid>
            </Grid>

            {!auditData.enhanced_data_available && (
              <Box sx={{ mt: 3, p: 3, bgcolor: 'info.light', borderRadius: 2 }}>
                <Typography variant="h6" color="info.dark" gutterBottom>
                  🔧 Enhanced Analysis Required
                </Typography>
                <Typography variant="body2" color="info.dark" paragraph>
                  {auditData.message || 'This session was analyzed with the legacy system. For detailed gaze direction analysis, please re-analyze with the enhanced attention tracking system.'}
                </Typography>
                <Typography variant="body2" color="info.dark" paragraph>
                  <strong>To get detailed audit data:</strong>
                </Typography>
                <Typography variant="body2" color="info.dark" component="div">
                  1. Run enhanced analysis: <code>python run_enhanced_analysis_test3.py</code><br/>
                  2. Or upload a new video using the enhanced analysis endpoint<br/>
                  3. The enhanced system provides:
                  <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                    <li>Detailed gaze direction breakdown (forward, down, left, right, up)</li>
                    <li>Frame-by-frame evidence with timestamps</li>
                    <li>Confidence scores and quality metrics</li>
                    <li>Joint attention detection</li>
                    <li>Visual tracking patterns</li>
                    <li>Complete audit trails for clinical authenticity</li>
                  </ul>
                </Typography>
              </Box>
            )}
          </Box>
        )}

        {/* Tab 2: Methodology */}
        {activeTab === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Analysis Methodology & Formulas
            </Typography>
            
            {/* Calculation Details */}
            {auditData.calculation_details && (
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom color="primary">
                  📊 Detailed Calculations
                </Typography>
                
                {/* Eye Contact Percentage */}
                {auditData.calculation_details.eye_contact_percentage && (
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Eye Contact Percentage Calculation
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1, mb: 1 }}>
                        {auditData.calculation_details.eye_contact_percentage.formula}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {auditData.calculation_details.eye_contact_percentage.calculation}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        <strong>Result:</strong> {auditData.calculation_details.eye_contact_percentage.result?.toFixed(2)}%
                      </Typography>
                    </CardContent>
                  </Card>
                )}
                
                {/* Average Confidence */}
                {auditData.calculation_details.average_confidence && (
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Average Confidence Calculation
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1, mb: 1 }}>
                        {auditData.calculation_details.average_confidence.formula}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {auditData.calculation_details.average_confidence.calculation}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        <strong>Result:</strong> {auditData.calculation_details.average_confidence.result?.toFixed(3)}
                      </Typography>
                    </CardContent>
                  </Card>
                )}
                
                {/* Duration Calculation */}
                {auditData.calculation_details.session_duration && (
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Session Duration Calculation
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1, mb: 1 }}>
                        {auditData.calculation_details.session_duration.formula}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {auditData.calculation_details.session_duration.calculation}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        <strong>Result:</strong> {auditData.calculation_details.session_duration.result?.toFixed(2)} seconds
                      </Typography>
                    </CardContent>
                  </Card>
                )}
                
                {/* Gaze Direction Classification Rules */}
                {auditData.calculation_details.gaze_direction_method && (
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Gaze Direction Classification Rules
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Each frame's gaze direction is classified using the following mathematical rules:
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell><strong>Direction</strong></TableCell>
                              <TableCell><strong>Mathematical Rule</strong></TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(auditData.calculation_details.gaze_direction_method).map(([direction, rule]) => (
                              <TableRow key={direction}>
                                <TableCell>
                                  {direction.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </TableCell>
                                <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                  {rule}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                )}
              </Box>
            )}

            {/* Methodology */}
            {auditData.methodology && (
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom color="primary">
                  🔬 Implementation Methods
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(auditData.methodology).map(([key, value]) => (
                    <Grid item xs={12} key={key}>
                      <Typography variant="subtitle2" color="primary">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ 
                        fontFamily: key.includes('formula') ? 'monospace' : 'inherit',
                        bgcolor: key.includes('formula') ? 'grey.100' : 'transparent',
                        p: key.includes('formula') ? 1 : 0,
                        borderRadius: key.includes('formula') ? 1 : 0
                      }}>
                        {value}
                      </Typography>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}

            {/* Detection Thresholds */}
            {auditData.thresholds_used && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom color="primary">
                  ⚙️ Detection Thresholds
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Parameter</strong></TableCell>
                        <TableCell><strong>Value</strong></TableCell>
                        <TableCell><strong>Purpose</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(auditData.thresholds_used).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell>
                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </TableCell>
                          <TableCell>
                            {typeof value === 'number' ? value.toFixed(3) : value}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {getThresholdPurpose(key)}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
}

export default EyeContactAuditSimple;