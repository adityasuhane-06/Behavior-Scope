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
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';
import SettingsIcon from '@mui/icons-material/Settings';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import InfoIcon from '@mui/icons-material/Info';
import { fetchResponsivenessAudit } from '../api/auditAPI';

/**
 * ResponsivenessAudit - Detailed audit trail for question-response analysis
 * Shows all detected questions, matched responses, calculation breakdown, and thresholds
 */
function ResponsivenessAudit() {
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
      const data = await fetchResponsivenessAudit(sessionId);
      setAuditData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load responsiveness audit');
    } finally {
      setLoading(false);
    }
  };

  const getMethodColor = (method) => {
    switch (method) {
      case 'semantic': return 'primary';
      case 'pitch_rise': return 'secondary';
      case 'hybrid': return 'success';
      default: return 'default';
    }
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

  const { calculation_audit, responsiveness_index, interpretation } = auditData;
  const { questions_detected, responses_matched, score_calculation, thresholds } = calculation_audit;

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
      <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #2196f3 0%, #1565c0 100%)' }}>
        <Typography variant="h4" color="white" gutterBottom>
          Question-Response Audit Trail
        </Typography>
        <Typography variant="body1" color="rgba(255,255,255,0.9)">
          Complete transparency into how the Responsiveness Index was calculated
        </Typography>
      </Paper>

      {/* Score Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Responsiveness Index
              </Typography>
              <Typography variant="h2" color="primary">
                {responsiveness_index?.toFixed(1) || score_calculation?.responsiveness_index?.toFixed(1) || 'N/A'}
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
                Questions Detected
              </Typography>
              <Typography variant="h2" color="info.main">
                {score_calculation?.total_questions || questions_detected?.length || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                from clinician
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Response Rate
              </Typography>
              <Typography variant="h2" color="success.main">
                {score_calculation?.response_rate?.toFixed(1) || '0.0'}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                questions answered
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
                {score_calculation?.mean_latency?.toFixed(2) || '0.00'}s
              </Typography>
              <Typography variant="body2" color="textSecondary">
                response time
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

      {/* Questions Table */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <QuestionAnswerIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Detected Questions ({questions_detected?.length || 0})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer sx={{ maxHeight: 400 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell><strong>#</strong></TableCell>
                  <TableCell><strong>Start Time</strong></TableCell>
                  <TableCell><strong>Duration</strong></TableCell>
                  <TableCell><strong>Response?</strong></TableCell>
                  <TableCell><strong>Latency</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {questions_detected?.map((q, idx) => (
                  <TableRow key={idx} hover>
                    <TableCell>{idx + 1}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {q.start_time?.toFixed(2)}s
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {q.duration?.toFixed(2)}s
                    </TableCell>
                    <TableCell>
                      {q.has_response ? (
                        <Chip label="Yes" size="small" color="success" />
                      ) : (
                        <Chip label="No" size="small" color="error" variant="outlined" />
                      )}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {q.response_latency ? `${q.response_latency.toFixed(2)}s` : '-'}
                    </TableCell>
                  </TableRow>
                ))}
                {(!questions_detected || questions_detected.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={5} align="center">
                      <Typography color="textSecondary">No questions detected</Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>

      {/* Responses Table */}
      <Accordion defaultExpanded sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <QuestionAnswerIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Response Matching ({responses_matched?.filter(r => r.answered).length || 0} answered)
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer sx={{ maxHeight: 400 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Q#</strong></TableCell>
                  <TableCell><strong>Question Time</strong></TableCell>
                  <TableCell><strong>Question</strong></TableCell>
                  <TableCell><strong>Response Time</strong></TableCell>
                  <TableCell><strong>Response</strong></TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <strong>Latency</strong>
                      <Tooltip title="Time elapsed between the end of the question and the start of the response." arrow placement="top">
                        <InfoIcon fontSize="inherit" sx={{ opacity: 0.5, cursor: 'help' }} />
                      </Tooltip>
                    </Box>
                  </TableCell>
                  <TableCell><strong>Answered</strong></TableCell>
                  <TableCell><strong>Appropriate</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {responses_matched?.map((row, idx) => (
                  <TableRow key={idx} hover sx={{ 
                    backgroundColor: row.answered ? 'inherit' : 'action.hover' 
                  }}>
                    <TableCell>{idx + 1}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {row.question_timestamp?.toFixed(2)}s
                    </TableCell>
                    <TableCell sx={{ fontWeight: 'medium', color: 'primary.main', maxWidth: 250 }}>
                      {row.question_text || '-'}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {row.response_timestamp ? `${row.response_timestamp.toFixed(2)}s` : '-'}
                    </TableCell>
                    <TableCell sx={{ maxWidth: 250 }}>
                      {row.response_text || '-'}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontWeight: row.latency > 3 ? 'bold' : 'regular', color: row.latency > 3 ? 'warning.main' : 'text.primary' }}>
                      {row.latency ? `${row.latency.toFixed(2)}s` : '-'}
                    </TableCell>
                    <TableCell>
                      {row.answered ? (
                        <Chip label="Yes" size="small" color="success" />
                      ) : (
                        <Chip label="No" size="small" color="error" variant="outlined" />
                      )}
                    </TableCell>
                    <TableCell>
                      {row.appropriate !== null ? (
                        row.appropriate ? (
                          <CheckCircleIcon color="success" fontSize="small" />
                        ) : (
                          <CancelIcon color="warning" fontSize="small" />
                        )
                      ) : (
                        <Typography color="textSecondary">-</Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
                {(!responses_matched || responses_matched.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      <Typography color="textSecondary">No response data</Typography>
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
            <strong>Formula:</strong> {score_calculation?.formula || 'Weighted: 0.5*response_rate + 0.3*appropriateness + 0.2*latency_score'}
          </Alert>
          
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell><strong>Total Questions</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.total_questions}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Answered Count</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.answered_count}
                  </TableCell>
                </TableRow>
                <TableRow sx={{ backgroundColor: 'grey.50' }}>
                  <TableCell><strong>Response Rate</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    ({score_calculation?.answered_count} / {score_calculation?.total_questions}) × 100 = <strong>{score_calculation?.response_rate?.toFixed(1)}%</strong>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Mean Response Latency</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.mean_latency?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Median Response Latency</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.median_latency?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Mean Response Duration</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.mean_duration?.toFixed(2)}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Appropriate Responses</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.appropriate_count}
                  </TableCell>
                </TableRow>
                <TableRow sx={{ backgroundColor: 'grey.50' }}>
                  <TableCell><strong>Appropriateness Rate</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {score_calculation?.appropriateness_rate?.toFixed(1)}%
                  </TableCell>
                </TableRow>
                <Divider />
                <TableRow sx={{ backgroundColor: 'primary.light' }}>
                  <TableCell><strong>Final Responsiveness Index</strong></TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace', fontWeight: 'bold', color: 'white' }}>
                    {score_calculation?.responsiveness_index?.toFixed(1)} / 100
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
                  <TableCell>Maximum Normal Latency</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.max_normal_latency_sec || 3.0}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Minimum Pitch Rise Threshold</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.min_pitch_rise_threshold || 0.15}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Minimum Question Duration</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.min_question_duration_sec || 0.5}s
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Semantic Confidence Threshold</TableCell>
                  <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                    {thresholds?.semantic_confidence_threshold || 0.7}
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

export default ResponsivenessAudit;
