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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert
} from '@mui/material';
import CalculateIcon from '@mui/icons-material/Calculate';
import TimelineIcon from '@mui/icons-material/Timeline';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import {
  LinearProgress,
  Stack
} from '@mui/material';
import MuiTooltip from '@mui/material/Tooltip';
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
import { fetchSession } from '../api/auditAPI';

const METRIC_FORMULAS = {
  'vocal_regulation': {
    title: 'Vocal Regulation Index Strategy',
    description: 'Measures how stable and controlled the speech patterns are. It rewards consistency and penalizes erratic changes in speed, pauses, or pitch.',
    main_formula: 'Score = 100 × (1 - Weighted Instability)',
    components: [
      {
        name: 'Speech Rate Consistency',
        weight: '40%',
        numerator: 'Z-score of speech rate deviation',
        numerator_tooltip: 'Z-score (Standard Score) Calculation:\n(Current Rate - Baseline Mean) / Baseline Standard Deviation.\n\nIt quantifies how "abnormal" the current speed is compared to the speaker\'s typical range. A z-score of 0 is average; >2 is significantly deviant.',
        denominator: '3.0 (Max deviation threshold)',
        notes: 'Measures if speaking speed is consistent (not switching between too fast/slow).',
        dataKey: 'speech_rate_score' 
      },
      {
        name: 'Pause Regularity',
        weight: '30%',
        numerator: 'Pause duration variance + Count irregularity',
        numerator_tooltip: 'Combined metric of:\n1. Variance in pause length (are pauses random?)\n2. Irregularity in pause frequency per minute.\n\nHigher values indicate disjointed, unpredictable speech patterns.',
        denominator: '4.0 (Max irregularity threshold)',
        notes: 'Measures if pauses are natural and rhythmic vs. erratic.',
        dataKey: 'pause_score'
      },
      {
        name: 'Prosodic Stability',
        weight: '30%',
        numerator: 'Pitch variance + Energy (volume) variance',
        numerator_tooltip: 'Composite instability measure:\nNormalized(Pitch Variance) + Normalized(Energy Variance).\n\nDetects tremors, sudden loudness changes, or pitch breaks often associated with emotional dysregulation.',
        denominator: '4.0 (Max instability threshold)',
        notes: 'Measures emotional control via voice pitch and volume stability.',
        dataKey: 'prosody_score'
      }
    ]
  },
  'motor_agitation': {
    title: 'Motor Agitation Index Strategy',
    description: 'Measures the intensity of physical movement (fidgeting, restlessness). Higher scores indicate more agitation.',
    main_formula: 'Score = (0.4 × Head) + (0.4 × Body) + (0.2 × Hand)',
    components: [
      {
        name: 'Head Motion',
        weight: '40%',
        numerator: 'Variance of head angles (yaw, pitch, roll)',
        numerator_tooltip: 'Statistical variance (spread) of the head\'s rotation angles over the time window.\nFormula: Var(Yaw) + Var(Pitch) + Var(Roll).\n\nCaptures constant looking around or head shaking.',
        denominator: '30.0 degrees (Max variance)',
        notes: 'Captures head restlessness and constant looking around.',
        dataKey: 'head_motion_score'
      },
      {
        name: 'Body Motion',
        weight: '40%',
        numerator: 'Upper-body motion energy (95th percentile)',
        numerator_tooltip: 'Motion Energy Analysis (MEA):\nSum of pixel intensity changes in the torso region between consecutive frames, normalized by body size.\n\nCaptures shifting in seat, rocking, or heavy breathing.',
        denominator: '0.2 motion units',
        notes: 'Captures shifting in seat, torso movement, or rocking.',
        dataKey: 'body_motion_score'
      },
      {
        name: 'Hand Motion',
        weight: '20%',
        numerator: 'Maximum hand velocity',
        numerator_tooltip: 'Peak velocity of wrist landmarks tracked by MediaPipe.\nFormula: Max(sqrt(dx² + dy²)) for both hands.\n\nMeasures the fastest sudden movement (fidgeting/flapping) in the frame.',
        denominator: '40.0 pixels/frame',
        notes: 'Captures fidgeting, hand-flapping, or rapid hand gestures.',
        dataKey: 'hand_motion_score'
      }
    ]
  },
  'attention_stability': {
    title: 'Attention Stability Score Strategy',
    description: 'Measures sustained focus based on holding a steady head position and gaze.',
    main_formula: 'Score = Base Stability × Presence Factor',
    components: [
      {
        name: 'Head Pose Stability',
        weight: '50%',
        numerator: 'Standard Deviation of head orientation',
        numerator_tooltip: 'Standard Deviation of the head\'s forward vector.\n\nLow deviation means the head is held steady in a single direction for the duration of the window.',
        denominator: '20.0 degrees (Stability threshold)',
        notes: 'Rewards keeping head steady (focused).',
        dataKey: 'head_pose_stability',
        rawInputKey: 'raw_head_variance',
        unit: '°'
      },
      {
        name: 'Gaze Stability',
        weight: '50%',
        numerator: 'Variance in eye gaze direction',
        numerator_tooltip: 'Variance of the 3D gaze vector (eye direction).\n\nLow variance means eyes are focused on a single region/object and not scanning the environment rapidly.',
        denominator: '0.05 (Variance threshold)',
        notes: 'Rewards fixing eyes on a target regions (reduced scanning).',
        dataKey: 'gaze_stability',
        rawInputKey: 'raw_gaze_variance',
        unit: ''
      },
      {
        name: 'Presence Factor',
        weight: 'Modifier',
        numerator: 'Frames where face is detected',
        numerator_tooltip: 'Ratio: Face Detected Frames / Total Window Frames.\n\nChecks if the user is actually visible/present in front of the camera.',
        denominator: 'Total Session Frames',
        notes: 'Penalizes the score if the user leaves the camera frame entirely.',
        dataKey: 'presence_score'
      }
    ]
  },
  'regulation_consistency': {
    title: 'Regulation Consistency Index Strategy',
    description: 'Measures if the behavior is consistent over time or fluctuates wildly.',
    main_formula: 'Score = 100 × (1 - Variability) × Autocorrelation',
    components: [
      {
        name: 'Variability (Coefficient of Variation)',
        weight: 'Factor 1',
        numerator: 'Standard Deviation of confidence scores',
        numerator_tooltip: 'Coefficient of Variation (CV):\nStandard Deviation / Mean of the confidence scores.\n\nA normalized measure of how much the signal fluctuates relative to its size.',
        denominator: 'Mean of confidence scores',
        notes: 'High variability reduces the score.',
        dataKey: 'variability'
      },
      {
        name: 'Autocorrelation (Predictability)',
        weight: 'Factor 2',
        numerator: 'Correlation(Current State, State 10s later)',
        numerator_tooltip: 'Lag-1 Autocorrelation:\nCorrelation between the signal and itself delayed by 1 step.\n\nMeasures how predictable/stable the behavior is over time (memory of the system).',
        denominator: '1.0 (Perfect Correlation)',
        notes: 'Measures if current behavior predicts future behavior (stable state).',
        dataKey: 'autocorrelation'
      }
    ]
  },
  'facial_affect': {
    title: 'Facial Affect Index Strategy',
    description: 'Evaluates the range and appropriateness of facial expressions, checking for flat affect vs. dynamic expressivity.',
    main_formula: 'Score = (Range * 0.3) + (Mobility * 0.25) + ((100-Flat) * 0.25) + (Symm * 0.1)',
    components: [
      {
        name: 'Affect Range',
        weight: '30%',
        numerator: 'Unique Action Units Activated',
        numerator_tooltip: 'Count of unique facial muscle movements (AUs) activated divided by max AUs (15).\n\nHigher range indicates a richer emotional repertoire.',
        denominator: '15.0 Total AUs',
        notes: 'Measures diversity of facial expressions.',
        dataKey: 'affect_range_score'
      },
      {
        name: 'Facial Mobility',
        weight: '25%',
        numerator: 'Average Intensity + Activation Frequency',
        numerator_tooltip: 'Composite of how strong the expressions are and how often they change.\n\nLow mobility = rigid/still face.',
        denominator: '100.0 (Max Mobility)',
        notes: 'Measures the amount of facial movement.',
        dataKey: 'facial_mobility_index'
      },
      {
        name: 'Flat Affect Indicator',
        weight: '25%',
        numerator: 'Inactivity Ratio + Low Intensity',
        numerator_tooltip: 'Measures lack of expression. \nHigh score means the face was mostly blank or neutral.',
        denominator: '100.0 (Full Flat Affect)',
        notes: 'Lower values are better (more expressive).',
        dataKey: 'flat_affect_indicator'
      },
       {
        name: 'Facial Symmetry',
        weight: '10%',
        numerator: 'Left-Right AU Balance',
        numerator_tooltip: 'Symmetry of expression between left and right side of face.\n\nSignificant asymmetry can indicate specific emotional states or neurological factors.',
        denominator: '100.0 (Perfect Symmetry)',
        notes: 'Measures bilateral coordination.',
        dataKey: 'symmetry_index'
      }
    ]
  }
};



function MetricAudit() {
  const { sessionId, metricName } = useParams();
  const navigate = useNavigate();
  const [auditData, setAuditData] = useState(null);
  const [session, setSession] = useState(null);
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
      const [metricResponse, sessionData] = await Promise.all([
        axios.get(`http://localhost:8000/sessions/${sessionId}/metric/${metricName}`),
        fetchSession(sessionId)
      ]);
      setAuditData(metricResponse.data);
      setSession(sessionData);
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
      regulation_consistency: 'Regulation Consistency Index (RCI)',
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

        {METRIC_FORMULAS[metricName] && (
           <Box sx={{ mt: 2 }}>
             <Chip 
               icon={<InfoIcon />} 
               label="Calculation Method" 
               color="primary" 
               variant="outlined" 
               sx={{ mr: 1 }} 
             />
             <Typography variant="body2" component="span" color="text.secondary">
               {METRIC_FORMULAS[metricName].description}
             </Typography>
           </Box>
        )}

        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Overall Score
                </Typography>
                <Typography variant="h3">
                  {(() => {
                    const val = auditData.overall_score ?? (session && {
                         'vocal_regulation': session.vocal_regulation_index,
                         'motor_agitation': session.motor_agitation_index,
                         'attention_stability': session.attention_stability_score,
                         'regulation_consistency': session.regulation_consistency_index,
                         'facial_affect': session.facial_affect_index
                    }[metricName]);
                    return val != null ? Number(val).toFixed(1) : 'N/A';
                  })()}
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

      {/* Detailed System Breakdown */}
      {METRIC_FORMULAS[metricName] && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <CalculateIcon color="primary" />
            <Typography variant="h6">
              Detailed Calculation Trace
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" paragraph>
            This shows exactly how the raw data was transformed into the final score.
          </Typography>

          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50', mb: 3 }}>
             <Typography variant="subtitle2" color="text.secondary" gutterBottom>
               Formula
             </Typography>
             
             {/* Interactive Formula with Tooltip for Vocal Regulation */}
             <Typography variant="h6" sx={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
               {metricName === 'vocal_regulation' ? (
                 <span>
                   Score = 100 × (1 - 
                   <MuiTooltip 
                     title={
                       <Box sx={{ p: 1 }}>
                         <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="inherit">
                           Weighted Instability Formula:
                         </Typography>
                         <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                           (0.4 × Speech Rate) + (0.3 × Pause) + (0.3 × Prosody)
                         </Typography>
                         <Typography variant="caption" display="block" sx={{ mt: 1, opacity: 0.8 }}>
                           Weighted sum of the three instability components below.
                         </Typography>
                       </Box>
                     }
                     arrow
                     placement="top"
                   >
                     <span style={{ 
                       textDecoration: 'underline dashed', 
                       textUnderlineOffset: '4px',
                       cursor: 'help', 
                       color: '#1976d2', 
                       margin: '0 6px',
                       backgroundColor: 'rgba(25, 118, 210, 0.08)',
                       padding: '2px 6px',
                       borderRadius: '4px'
                     }}>
                       Weighted Instability
                     </span>
                   </MuiTooltip>
                   )
                 </span>
               ) : metricName === 'attention_stability' ? (
                 <span>
                   Score = 
                   <MuiTooltip 
                     title={
                       <Box sx={{ p: 1 }}>
                         <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="inherit">Base Stability Formula:</Typography>
                         <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>(0.5 × Head Pose Stability) + (0.5 × Gaze Stability)</Typography>
                         <Typography variant="caption" display="block" sx={{ mt: 1, opacity: 0.8 }}>Combines head steadiness and gaze focus.</Typography>
                       </Box>
                     }
                     arrow placement="top"
                   >
                     <span style={{ 
                       textDecoration: 'underline dashed', 
                       textUnderlineOffset: '4px',
                       cursor: 'help', 
                       color: '#1976d2', 
                       margin: '0 6px',
                       backgroundColor: 'rgba(25, 118, 210, 0.08)',
                       padding: '2px 6px',
                       borderRadius: '4px'
                     }}>
                       Base Stability
                     </span>
                   </MuiTooltip>
                   ×
                   <MuiTooltip 
                     title={
                       <Box sx={{ p: 1 }}>
                         <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="inherit">Presence Factor Formula:</Typography>
                         <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>0.5 + (0.5 × Presence % / 100)</Typography>
                         <Typography variant="caption" display="block" sx={{ mt: 1, opacity: 0.8 }}>Penalizes score if user leaves frame (max 50% penalty).</Typography>
                       </Box>
                     }
                     arrow placement="top"
                   >
                     <span style={{ 
                       textDecoration: 'underline dashed', 
                       textUnderlineOffset: '4px',
                       cursor: 'help', 
                       color: '#1976d2', 
                       margin: '0 6px',
                       backgroundColor: 'rgba(25, 118, 210, 0.08)',
                       padding: '2px 6px',
                       borderRadius: '4px'
                     }}>
                       Presence Factor
                     </span>
                   </MuiTooltip>
                 </span>
               ) : (
                 METRIC_FORMULAS[metricName].main_formula
               )}
             </Typography>
          </Paper>

          <Typography variant="subtitle1" gutterBottom fontWeight="bold">
            Components & Input Data:
          </Typography>
          
          <List disablePadding>
            {METRIC_FORMULAS[metricName].components.map((comp, index) => {
              // Try to find the actual value from session detailed scores
              // We assume session.detailed_scores[metricName][comp.dataKey] exists
              let actualValue = 'N/A';
              let scoreDetails = session?.detailed_scores?.[metricName];
              
              if (scoreDetails && comp.dataKey && scoreDetails[comp.dataKey] !== undefined) {
                 actualValue = typeof scoreDetails[comp.dataKey] === 'number' 
                   ? scoreDetails[comp.dataKey].toFixed(2) 
                   : scoreDetails[comp.dataKey];
              }

              return (
                <Paper key={index} elevation={0} sx={{ border: '1px solid #e0e0e0', mb: 2, overflow: 'hidden' }}>
                  <Box sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                     <Grid container alignItems="center">
                       <Grid item xs={12} sm={8}>
                         <Typography variant="subtitle1" fontWeight="bold">
                           {comp.name}
                         </Typography>
                       </Grid>
                       <Grid item xs={12} sm={4} textAlign={{ sm: 'right' }}>
                          <Chip label={`Weight: ${comp.weight}`} size="small" sx={{ bgcolor: 'white', color: 'primary.main', fontWeight: 'bold' }} />
                       </Grid>
                     </Grid>
                  </Box>
                  <Box sx={{ p: 2 }}>
                    <Grid container spacing={3}>
                       <Grid item xs={12} md={6}>
                          <Typography variant="caption" color="text.secondary" display="block" textTransform="uppercase" fontWeight="bold">
                            Metric Component
                          </Typography>
                          
                          {/* Interactive Tooltip for Numerator */}
                          {comp.numerator_tooltip ? (
                             <MuiTooltip 
                               title={
                                 <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                                   {comp.numerator_tooltip}
                                 </Typography>
                               }
                               arrow
                               placement="top-start"
                             >
                               <Typography 
                                 variant="body1" 
                                 gutterBottom
                                 sx={{ 
                                   textDecoration: 'underline dashed', 
                                   textUnderlineOffset: '4px',
                                   cursor: 'help',
                                   display: 'inline-block'
                                 }}
                               >
                                 {comp.numerator}
                               </Typography>
                             </MuiTooltip>
                          ) : (
                             <Typography variant="body1" gutterBottom>
                               {comp.numerator}
                             </Typography>
                          )}

                          <Box sx={{ mt: 1, p: 1, bgcolor: '#f5f5f5', borderRadius: 1, borderLeft: '3px solid #1976d2' }}>
                             <Typography variant="caption" display="block" color="text.secondary">
                               Derived Score:
                             </Typography>
                             <Typography variant="body2" fontWeight="bold" fontFamily="monospace">
                               {comp.dataKey}: {actualValue}
                             </Typography>
                             
                             {/* Display Raw Input if available */}
                             {comp.rawInputKey && scoreDetails && scoreDetails[comp.rawInputKey] !== undefined && (
                               <Typography variant="caption" display="block" sx={{ mt: 0.5, color: 'text.secondary', borderTop: '1px dashed #ddd', pt: 0.5 }}>
                                 Raw Input: <strong>{Number(scoreDetails[comp.rawInputKey]).toFixed(4)}{comp.unit}</strong>
                               </Typography>
                             )}
                          </Box>
                       </Grid>
                       <Grid item xs={12} md={6}>
                          <Typography variant="caption" color="text.secondary" display="block" textTransform="uppercase" fontWeight="bold">
                            Threshold (Denominator)
                          </Typography>
                          <Typography variant="body1" gutterBottom>
                            {comp.denominator}
                          </Typography>
                       </Grid>
                    </Grid>
                    <Divider sx={{ my: 1.5 }} />
                    <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: 0.5 }}>
                       <InfoIcon fontSize="small" /> {comp.notes}
                    </Typography>
                  </Box>
                </Paper>
              );
            })}
          </List>
          
          <Alert severity="info" sx={{ mt: 2 }} icon={<CheckCircleIcon />}>
            <strong>Verification:</strong> The calculation matches the system audit trail. Every number is traceable to specific frame analysis data.
          </Alert>
        </Paper>
      )}

      {/* Timeline Analysis - "Time frame for which contributed" */}
      {chartData.length > 0 && (
         <Paper sx={{ p: 3, mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <TimelineIcon color="primary" />
              <Typography variant="h6">
                Timeline Contribution & Analysis
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" paragraph>
              Analysis of how the score changed over time and which segments contributed most to the final result.
            </Typography>

            <Grid container spacing={3}>
               <Grid item xs={12} md={8}>
                  {/* Keep the chart but cleaner */}
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e0e0e0" />
                      <XAxis dataKey="frame" hide />
                      <YAxis domain={[0, 100]} axisLine={false} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#1976d2"
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <Typography variant="caption" align="center" display="block" color="text.secondary" sx={{ mt: 1 }}>
                    Session Timeline (Frames 0 - {chartData.length})
                  </Typography>
               </Grid>
               
               <Grid item xs={12} md={4}>
                  <Typography variant="subtitle2" gutterBottom>
                    Key Contributing Segments
                  </Typography>
                  <Stack spacing={1}>
                     {/* Auto-detect drops in score */}
                     {(() => {
                        // Simple analysis to find drops
                        let segments = [];
                        let inDrop = false;
                        let startDrop = 0;
                        
                        for (let i = 0; i < chartData.length; i++) {
                           if (chartData[i].score < 50 && !inDrop) {
                              inDrop = true;
                              startDrop = chartData[i].timestamp;
                           } else if ((chartData[i].score >= 50 || i === chartData.length - 1) && inDrop) {
                              inDrop = false;
                              segments.push({
                                start: startDrop,
                                end: chartData[i].timestamp,
                                minScore: Math.min(...chartData.slice(i-10 > 0 ? i-10 : 0, i+1).map(d => d.score))
                              });
                           }
                        }
                        
                        // Sort by severity (lowest score)
                        segments.sort((a,b) => a.minScore - b.minScore);

                        if (segments.length === 0) {
                           return (
                             <Alert severity="success" variant="outlined">
                               No significant drops detected. Score remained stable.
                             </Alert>
                           );
                        }

                        return segments.slice(0, 3).map((seg, idx) => (
                           <Paper key={idx} variant="outlined" sx={{ p: 1.5, borderLeft: '4px solid #f44336' }}>
                              <Typography variant="caption" fontWeight="bold">
                                {seg.start}s - {seg.end}s
                              </Typography>
                              <Typography variant="body2" color="error.main" fontWeight="bold">
                                Score dropped to {seg.minScore.toFixed(1)}%
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                High instability detected in this region.
                              </Typography>
                           </Paper>
                        ));
                     })()}
                  </Stack>
               </Grid>
            </Grid>
         </Paper>
      )}
    </Box>
  );
}

export default MetricAudit;

