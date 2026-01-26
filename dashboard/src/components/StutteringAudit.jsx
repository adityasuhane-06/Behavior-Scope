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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Alert,
  Tooltip
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CalculateIcon from '@mui/icons-material/Calculate';
import TimelineIcon from '@mui/icons-material/Timeline';
import SettingsIcon from '@mui/icons-material/Settings';
import InfoIcon from '@mui/icons-material/Info';
import { fetchStutteringAudit } from '../api/auditAPI';

/**
 * StutteringAudit - Detailed audit trail for stuttering/disfluency analysis
 * Shows all detected events, calculation breakdown, and thresholds used
 */
function StutteringAudit() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [auditData, setAuditData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadAuditData();
  }, [sessionId]);

  const loadAuditData = async () => {
    try {
      setLoading(true);
      const data = await fetchStutteringAudit(sessionId);
      setAuditData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load stuttering audit');
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'mild': return 'success';
      case 'moderate': return 'warning';
      case 'severe': return 'error';
      default: return 'default';
    }
  };

  const getSourceColor = (source) => {
    return source === 'transcript' ? 'primary' : 'secondary';
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
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(`/session/${sessionId}`)}
        >
          Back to Session
        </Button>
      </Box>
    );
  }

  if (!auditData || !auditData.calculation_audit) {
    return (
      <Box textAlign="center" mt={4}>
        <Alert severity="info">No audit data available for this session.</Alert>
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

  const { calculation_audit, stuttering_severity_index, interpretation } = auditData;
  const { events_detected, score_calculation, thresholds, type_distribution, severity_distribution } = calculation_audit;

  return (
    <Box>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(`/session/${sessionId}`)}
        sx={{ mb: 2 }}
      >
        Back to Session
      </Button>

      {/* Header */}
      <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)' }}>
        <Typography variant="h4" color="white" gutterBottom>
          Stuttering/Disfluency Audit Trail
        </Typography>
        <Typography variant="body1" color="rgba(255,255,255,0.9)">
          Complete transparency into how the Stuttering Severity Index (SSI) was calculated
        </Typography>
      </Paper>

      {/* Score Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Stuttering Severity Index
              </Typography>
              <Typography variant="h2" color="warning.main">
                {stuttering_severity_index?.toFixed(1) || score_calculation?.final_ssi?.toFixed(1) || 'N/A'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                /100 (higher = more severe)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Disfluencies
              </Typography>
              <Typography variant="h2" color="primary">
                {score_calculation?.total_disfluencies || events_detected?.length || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                events detected
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                Disfluency Rate
                <Tooltip 
                  title={
                    <Box sx={{ p: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, color: '#fff' }}>How is this calculated?</Typography>
                      <Typography variant="body2" sx={{ mb: 1, color: 'rgba(255,255,255,0.9)' }}>
                        <strong>Formula:</strong> (Total Disfluencies / Estimated Syllables) × 100
                      </Typography>
                      <Divider sx={{ my: 1, borderColor: 'rgba(255,255,255,0.3)' }} />
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 0.5, color: '#fff' }}>Why over 100%?</Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.9)' }}>
                        This happens when there are <strong>more stutter events than syllables</strong>.
                        <br /><br />
                        <strong>Example:</strong> Saying "b-b-b-ball" takes ~0.5s (estimated 1.5 syllables) but has 3 stutter events.
                        <br />
                        Rate = 3 / 1.5 = <strong>200%</strong>.
                      </Typography>
                    </Box>
                  }
                  arrow
                  placement="top"
                  componentsProps={{
                    tooltip: {
                      sx: {
                        bgcolor: 'rgba(0, 0, 0, 0.95)',
                        maxWidth: 400,
                        fontSize: '0.9rem',
                        border: '1px solid rgba(255,255,255,0.2)'
                      }
                    }
                  }}
                >
                  <InfoIcon fontSize="small" sx={{ cursor: 'help', color: 'action.active' }} />
                </Tooltip>
              </Typography>
              <Typography variant="h2" color="secondary">
                {score_calculation?.disfluency_rate_per_100?.toFixed(1) || '0.0'}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                per 100 syllables
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Interpretation */}
      {interpretation && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <strong>Clinical Interpretation:</strong> {interpretation}
        </Alert>
      )}

      {/* Events Table */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <TimelineIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Detected Disfluency Events ({events_detected?.length || 0})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer>
            <Table size="small">
              <TableHead>
                  <TableRow sx={{ backgroundColor: 'grey.100' }}>
                  <TableCell><strong>#</strong></TableCell>
                  <TableCell><strong>Timestamp</strong></TableCell>
                  <TableCell><strong>Type</strong></TableCell>
                  <TableCell><strong>Word / Context</strong></TableCell>
                  <TableCell><strong>Duration</strong></TableCell>
                  <TableCell><strong>Severity</strong></TableCell>
                  <TableCell><strong>Confidence</strong></TableCell>
                  <TableCell><strong>Details</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {events_detected?.map((event, idx) => (
                  <TableRow key={idx} hover>
                    <TableCell>{idx + 1}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {event.start_time?.toFixed(2)}s
                    </TableCell>
                    <TableCell>
                      <Chip label={event.disfluency_type} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell sx={{ fontWeight: 'medium', color: 'primary.main' }}>
                      {event.associated_word || '-'}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {event.duration?.toFixed(2)}s
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={event.severity} 
                        size="small" 
                        color={getSeverityColor(event.severity)}
                      />
                    </TableCell>
                    <TableCell>
                      {(event.confidence * 100).toFixed(0)}%
                    </TableCell>
                    <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {event.audio_pattern || '-'}
                    </TableCell>
                  </TableRow>
                ))}
                {(!events_detected || events_detected.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography color="textSecondary">No disfluency events detected</Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>

      {/* Calculation Breakdown */}
      <Accordion defaultExpanded sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <CalculateIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Score Calculation Breakdown</Typography>
        </AccordionSummary>
        <AccordionDetails>
            <Alert severity="info" sx={{ mb: 2 }}>
              <strong>Formula:</strong> {score_calculation?.formula || 'SSI = Frequency Score (0-40) + Duration Score (0-40) + Severity Score (0-20)'}
            </Alert>
          
          {/* SSI Component Breakdown */}
          <Paper variant="outlined" sx={{ p: 2, mb: 2, backgroundColor: 'warning.50' }}>
            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
              SSI Score Components:
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Tooltip 
                  title={
                    <Box sx={{ p: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Frequency Score (0-40 points)</Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        Based on <strong>disfluency rate</strong> (stutters per 100 syllables):
                      </Typography>
                      <Typography variant="body2" component="div">
                        • Rate &lt; 1%: 0 points<br/>
                        • Rate 1-3%: 10 points<br/>
                        • Rate 3-5%: 20 points<br/>
                        • Rate 5-10%: 30 points<br/>
                        • Rate &gt; 10%: 40 points
                      </Typography>
                    </Box>
                  }
                  arrow
                  placement="top"
                >
                  <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'primary.50', borderRadius: 1, cursor: 'help' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                      Frequency Score <InfoIcon sx={{ fontSize: 12 }} />
                    </Typography>
                    <Typography variant="h5" color="primary.main">
                      {score_calculation?.frequency_score || 0}/{score_calculation?.frequency_max || 40}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Rate: {score_calculation?.disfluency_rate_per_100?.toFixed(1)}%
                    </Typography>
                  </Box>
                </Tooltip>
              </Grid>
              <Grid item xs={4}>
                <Tooltip 
                  title={
                    <Box sx={{ p: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Duration Score (0-40 points)</Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        Based on <strong>mean duration</strong> of disfluency events:
                      </Typography>
                      <Typography variant="body2" component="div">
                        • Duration &lt; 0.3s: 0 points<br/>
                        • Duration 0.3-0.6s: 10 points<br/>
                        • Duration 0.6-1.0s: 20 points<br/>
                        • Duration 1.0-2.0s: 30 points<br/>
                        • Duration &gt; 2.0s: 40 points
                      </Typography>
                    </Box>
                  }
                  arrow
                  placement="top"
                >
                  <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'secondary.50', borderRadius: 1, cursor: 'help' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                      Duration Score <InfoIcon sx={{ fontSize: 12 }} />
                    </Typography>
                    <Typography variant="h5" color="secondary.main">
                      {score_calculation?.duration_score || 0}/{score_calculation?.duration_max || 40}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Mean: {score_calculation?.mean_event_duration?.toFixed(2) || '0.00'}s
                    </Typography>
                  </Box>
                </Tooltip>
              </Grid>
              <Grid item xs={4}>
                <Tooltip 
                  title={
                    <Box sx={{ p: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Severity Score (0-20 points)</Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        Based on proportion of <strong>severe</strong> disfluency events.
                      </Typography>
                      <Typography variant="body2">
                        Score = (Severe Events / Total Events) × 40, capped at 20 points.
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
                        Higher severity = longer blocks, more struggle behavior.
                      </Typography>
                    </Box>
                  }
                  arrow
                  placement="top"
                >
                  <Box sx={{ textAlign: 'center', p: 1, backgroundColor: 'error.50', borderRadius: 1, cursor: 'help' }}>
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                      Severity Score <InfoIcon sx={{ fontSize: 12 }} />
                    </Typography>
                    <Typography variant="h5" color="error.main">
                      {score_calculation?.severity_score?.toFixed(1) || 0}/{score_calculation?.severity_max || 20}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Severe events
                    </Typography>
                  </Box>
                </Tooltip>
              </Grid>
            </Grid>
            <Divider sx={{ my: 2 }} />
            <Tooltip 
              title="The final SSI score is the sum of all three components. Higher scores indicate more severe stuttering (0-20: Mild, 20-40: Moderate, 40-60: Moderately Severe, 60-100: Severe)."
              arrow
              placement="top"
            >
              <Box sx={{ textAlign: 'center', cursor: 'help' }}>
                <Typography variant="body2" color="textSecondary">
                  {score_calculation?.frequency_score || 0} + {score_calculation?.duration_score || 0} + {score_calculation?.severity_score?.toFixed(1) || 0} = 
                </Typography>
                <Typography variant="h4" color="warning.main" sx={{ fontWeight: 'bold' }}>
                  {score_calculation?.final_ssi?.toFixed(1)} / 100
                </Typography>
              </Box>
            </Tooltip>
          </Paper>
          
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell>
                    <Tooltip title="Total time the patient was speaking during the session" arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Total Speaking Time</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.total_speaking_time?.toFixed(2)} seconds
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Tooltip title="Estimated number of syllables based on speaking time (typically 3 syllables per second)" arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Estimated Syllables</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.estimated_syllables?.toFixed(0)}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Tooltip title="Total number of detected stuttering events (repetitions, blocks, prolongations)" arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Total Disfluencies</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.total_disfluencies}
                  </TableCell>
                </TableRow>
                <TableRow sx={{ backgroundColor: 'grey.50' }}>
                  <TableCell>
                    <Tooltip title="Percentage of syllables affected by disfluency. Above 5% is clinically significant." arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Disfluency Rate (per 100 syllables)</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    ({score_calculation?.total_disfluencies} / {score_calculation?.estimated_syllables?.toFixed(0)}) × 100 = <strong>{score_calculation?.disfluency_rate_per_100?.toFixed(2)}%</strong>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Tooltip title="Duration of the longest blocking event (complete stoppage of speech). Longer blocks indicate more severe stuttering." arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Longest Block Duration</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.longest_block_duration?.toFixed(2)} seconds
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Tooltip title="Average number of repetition units (e.g., 'ba-ba-ba' = 3 units). More units indicate more severe repetitions." arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Average Repetition Units</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.average_repetition_units?.toFixed(1)}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Tooltip title="Average duration of all disfluency events. Used to calculate the Duration Score." arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Mean Event Duration</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.mean_event_duration?.toFixed(2) || '0.00'} seconds
                  </TableCell>
                </TableRow>
                <Divider />
                <TableRow sx={{ backgroundColor: 'warning.light' }}>
                  <TableCell>
                    <Tooltip title="Stuttering Severity Index (SSI-4 inspired). 0-20: Mild, 20-40: Moderate, 40-60: Mod-Severe, 60+: Severe" arrow>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'help' }}>
                        <strong>Final SSI Score</strong> <InfoIcon sx={{ fontSize: 14, color: 'action.active' }} />
                      </Box>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                    {score_calculation?.final_ssi?.toFixed(1)} / 100
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>

          {/* Type Distribution */}
          {type_distribution && Object.keys(type_distribution).length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>Type Distribution:</Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {Object.entries(type_distribution).map(([type, count]) => (
                  <Chip 
                    key={type} 
                    label={`${type}: ${count}`} 
                    variant="outlined"
                    size="small"
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Severity Distribution */}
          {severity_distribution && Object.keys(severity_distribution).length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Severity Distribution:</Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {Object.entries(severity_distribution).map(([severity, count]) => (
                  <Chip 
                    key={severity} 
                    label={`${severity}: ${count}`} 
                    color={getSeverityColor(severity)}
                    size="small"
                  />
                ))}
              </Box>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>

      {/* Thresholds Used */}
      <Accordion sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <SettingsIcon sx={{ mr: 1 }} />
          <Typography variant="h6">Thresholds & Parameters Used</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow sx={{ backgroundColor: 'grey.100' }}>
                  <TableCell><strong>Parameter</strong></TableCell>
                  <TableCell align="right"><strong>Value</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Minimum Repetition Count</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.min_repetition_count || 2}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Minimum Block Duration</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.min_block_duration_sec || 0.3}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Minimum Prolongation Duration</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.min_prolongation_duration_sec || 0.5}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Syllables per Second Estimate</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.syllables_per_second_estimate || 3.0}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}

export default StutteringAudit;
