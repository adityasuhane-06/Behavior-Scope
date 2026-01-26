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
  LinearProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CalculateIcon from '@mui/icons-material/Calculate';
import TimelineIcon from '@mui/icons-material/Timeline';
import SettingsIcon from '@mui/icons-material/Settings';
import PeopleIcon from '@mui/icons-material/People';
import { fetchTurnTakingAudit } from '../api/auditAPI';

/**
 * TurnTakingAudit - Detailed audit trail for turn-taking analysis
 * Shows all detected turns, calculation breakdown, and thresholds used
 */
function TurnTakingAudit() {
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
      const data = await fetchTurnTakingAudit(sessionId);
      setAuditData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load turn-taking audit');
    } finally {
      setLoading(false);
    }
  };

  const getSpeakerColor = (speaker) => {
    return speaker === 'child' ? 'primary' : 'secondary';
  };

  const getSourceColor = (source) => {
    return source === 'transcript' ? 'success' : 'default';
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

  const { calculation_audit, reciprocity_score, explanation } = auditData;
  const { turns_detected, score_calculation, thresholds } = calculation_audit;

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
      <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)' }}>
        <Typography variant="h4" color="white" gutterBottom>
          Turn-Taking Audit Trail
        </Typography>
        <Typography variant="body1" color="rgba(255,255,255,0.9)">
          Complete transparency into how conversational reciprocity was calculated
        </Typography>
      </Paper>

      {/* Score Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Reciprocity Score
              </Typography>
              <Typography variant="h2" color="success.main">
                {reciprocity_score?.toFixed(1) || score_calculation?.reciprocity_score?.toFixed(1) || 'N/A'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                /100 (higher = better)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Turns
              </Typography>
              <Typography variant="h2" color="primary">
                {score_calculation?.total_turns || turns_detected?.length || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                exchanges detected
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Child Percentage
              </Typography>
              <Typography variant="h2" color="info.main">
                {score_calculation?.child_percentage?.toFixed(1) || '0.0'}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                of speaking time
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Mean Latency
              </Typography>
              <Typography variant="h2" color="warning.main">
                {score_calculation?.mean_response_latency?.toFixed(2) || '0.00'}s
              </Typography>
              <Typography variant="body2" color="textSecondary">
                response time
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Balance Visualization */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <PeopleIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Speaker Balance
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2" sx={{ minWidth: 80 }}>Child</Typography>
          <Box sx={{ flexGrow: 1 }}>
            <LinearProgress 
              variant="determinate" 
              value={score_calculation?.child_percentage || 0} 
              sx={{ height: 20, borderRadius: 2 }}
            />
          </Box>
          <Typography variant="body2" sx={{ minWidth: 50 }}>
            {score_calculation?.child_percentage?.toFixed(1)}%
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
          <Typography variant="body2" sx={{ minWidth: 80 }}>Therapist</Typography>
          <Box sx={{ flexGrow: 1 }}>
            <LinearProgress 
              variant="determinate" 
              value={100 - (score_calculation?.child_percentage || 0)} 
              color="secondary"
              sx={{ height: 20, borderRadius: 2 }}
            />
          </Box>
          <Typography variant="body2" sx={{ minWidth: 50 }}>
            {(100 - (score_calculation?.child_percentage || 0)).toFixed(1)}%
          </Typography>
        </Box>
        <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
          Ideal balance: 50% each (deviation: {score_calculation?.balance_deviation_from_ideal?.toFixed(1)}%)
        </Typography>
      </Paper>

      {/* Interpretation */}
      {explanation && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <strong>Clinical Interpretation:</strong> {explanation}
        </Alert>
      )}

      {/* Turns Table */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <TimelineIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Detected Turns ({turns_detected?.length || 0})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer sx={{ maxHeight: 400 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell><strong>#</strong></TableCell>
                  <TableCell><strong>Speaker</strong></TableCell>
                  <TableCell><strong>Start</strong></TableCell>
                  <TableCell><strong>End</strong></TableCell>
                  <TableCell><strong>Duration</strong></TableCell>
                  <TableCell><strong>Latency</strong></TableCell>
                  <TableCell><strong>Interruption</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {turns_detected?.map((turn, idx) => (
                  <TableRow key={idx} hover>
                    <TableCell>{idx + 1}</TableCell>
                    <TableCell>
                      <Chip 
                        label={turn.speaker} 
                        size="small" 
                        color={getSpeakerColor(turn.speaker)}
                      />
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {turn.start_time?.toFixed(2)}s
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {turn.end_time?.toFixed(2)}s
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {turn.duration?.toFixed(2)}s
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {turn.response_latency ? `${turn.response_latency.toFixed(2)}s` : '-'}
                    </TableCell>
                    <TableCell>
                      {turn.interruption ? (
                        <Chip label="Yes" size="small" color="error" />
                      ) : (
                        <Typography color="textSecondary">-</Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
                {(!turns_detected || turns_detected.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography color="textSecondary">No turns detected</Typography>
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
            <strong>Formula:</strong> {score_calculation?.formula || 'Reciprocity = f(balance, latency, interruptions)'}
          </Alert>
          
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell><strong>Child Turns</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.child_turns}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Therapist Turns</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.therapist_turns}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Child Speaking Time</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.child_speaking_time?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Therapist Speaking Time</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.therapist_speaking_time?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow sx={{ backgroundColor: 'grey.50' }}>
                  <TableCell><strong>Child Percentage</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.child_percentage?.toFixed(1)}%
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Balance Deviation from Ideal (50%)</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.balance_deviation_from_ideal?.toFixed(1)}%
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Mean Response Latency</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.mean_response_latency?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Median Response Latency</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.median_response_latency?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Interruption Count</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.interruption_count}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Balance Score</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.balance_score?.toFixed(1)}
                  </TableCell>
                </TableRow>
                <Divider />
                <TableRow sx={{ backgroundColor: 'success.light' }}>
                  <TableCell><strong>Final Reciprocity Score</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
                    {score_calculation?.reciprocity_score?.toFixed(1)} / 100
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
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
                  <TableCell>Ideal Balance Percentage</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.ideal_balance_percentage || 50}%
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Typical Latency Threshold</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.typical_latency_sec || 1.0}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Elevated Latency Threshold</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.elevated_latency_sec || 2.0}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Interruption Threshold</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.interruption_threshold_sec || 0.5}s
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

export default TurnTakingAudit;
