import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Chip,
  Grid,
  Divider,
  Tabs,
  Tab,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import PsychologyIcon from '@mui/icons-material/Psychology';
import TranscribeIcon from '@mui/icons-material/Transcribe';
import axios from 'axios';

function TranscriptViewer({ sessionId }) {
  const [transcriptData, setTranscriptData] = useState(null);
  const [clinicalData, setClinicalData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [editedText, setEditedText] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    loadTranscript();
    loadClinicalTranscript();
  }, [sessionId]);

  const loadTranscript = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get(`http://localhost:8000/sessions/${sessionId}/transcript`);
      setTranscriptData(response.data);
      setEditedText(response.data.transcript_text || '');
    } catch (err) {
      if (err.response?.status === 404) {
        setError('No transcript available for this session');
      } else {
        setError('Failed to load transcript');
      }
      console.error('Failed to load transcript:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadClinicalTranscript = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/sessions/${sessionId}/clinical-transcript`);
      setClinicalData(response.data);
    } catch (err) {
      // Clinical transcript is optional
      console.log('No clinical transcript available');
    }
  };

  const handleEdit = () => {
    setEditing(true);
    setSuccess(false);
  };

  const handleCancel = () => {
    setEditing(false);
    setEditedText(transcriptData?.transcript_text || '');
    setSuccess(false);
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      
      await axios.put(`http://localhost:8000/sessions/${sessionId}/transcript`, {
        text: editedText
      });

      setTranscriptData(prev => ({
        ...prev,
        transcript_text: editedText
      }));
      
      setEditing(false);
      setSuccess(true);
      
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError('Failed to save transcript');
      console.error('Failed to save transcript:', err);
    } finally {
      setSaving(false);
    }
  };

  const parseTranscriptJson = (jsonString) => {
    try {
      return JSON.parse(jsonString);
    } catch {
      return [];
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

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error && !transcriptData) {
    return (
      <Box textAlign="center" py={4}>
        <Alert severity="info" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Typography variant="body2" color="textSecondary">
          This could happen if the audio quality was too low for transcription or if the speech segments were too short.
        </Typography>
      </Box>
    );
  }

  const segments = transcriptData?.transcript_json ? parseTranscriptJson(transcriptData.transcript_json) : [];

  return (
    <Box>
      <Paper sx={{ p: 3 }}>
        {/* Tabs for Basic vs Clinical Transcript */}
        {clinicalData && (
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)} sx={{ mb: 2 }}>
            <Tab icon={<TranscribeIcon />} label="Basic Transcript" />
            <Tab icon={<PsychologyIcon />} label="Clinical Analysis" />
          </Tabs>
        )}

        {/* Tab Panel 0: Basic Transcript */}
        {tabValue === 0 && (
          <Box>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Speech Transcript
              </Typography>
              <Box>
                {!editing ? (
                  <Button
                    startIcon={<EditIcon />}
                    onClick={handleEdit}
                    variant="outlined"
                    size="small"
                  >
                    Edit Transcript
                  </Button>
                ) : (
                  <Box display="flex" gap={1}>
                    <Button
                      startIcon={<SaveIcon />}
                      onClick={handleSave}
                      variant="contained"
                      size="small"
                      disabled={saving}
                    >
                      {saving ? 'Saving...' : 'Save'}
                    </Button>
                    <Button
                      startIcon={<CancelIcon />}
                      onClick={handleCancel}
                      variant="outlined"
                      size="small"
                      disabled={saving}
                    >
                      Cancel
                    </Button>
                  </Box>
                )}
              </Box>
            </Box>

            {success && (
              <Alert severity="success" sx={{ mb: 2 }}>
                Transcript updated successfully!
              </Alert>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {/* Transcript Statistics */}
            {transcriptData && (
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                    <Typography variant="h6">
                      {transcriptData.word_count || 0}
                    </Typography>
                    <Typography variant="body2">Total Words</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, bgcolor: 'secondary.light', color: 'secondary.contrastText' }}>
                    <Typography variant="h6">
                      {transcriptData.speaker_count || 0}
                    </Typography>
                    <Typography variant="body2">Speakers</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, bgcolor: 'info.light', color: 'info.contrastText' }}>
                    <Typography variant="h6">
                      {segments.length}
                    </Typography>
                    <Typography variant="body2">Segments</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Paper sx={{ p: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
                    <Typography variant="h6">
                      Gemini
                    </Typography>
                    <Typography variant="body2">ASR Model</Typography>
                  </Paper>
                </Grid>
              </Grid>
            )}

            <Divider sx={{ mb: 2 }} />

            {/* Editable Transcript */}
            {editing ? (
              <TextField
                multiline
                fullWidth
                rows={15}
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                placeholder="Edit the transcript here..."
                variant="outlined"
                sx={{ mb: 2 }}
              />
            ) : (
              <Box>
                {transcriptData?.transcript_text ? (
                  <Typography
                    variant="body1"
                    component="pre"
                    sx={{
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'monospace',
                      bgcolor: '#f5f5f5',
                      p: 2,
                      borderRadius: 1,
                      maxHeight: '400px',
                      overflow: 'auto'
                    }}
                  >
                    {transcriptData.transcript_text}
                  </Typography>
                ) : (
                  <Alert severity="info">
                    No transcript text available. The audio may not have contained clear speech.
                  </Alert>
                )}
              </Box>
            )}

            {/* Detailed Segments */}
            {segments.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Detailed Segments
                </Typography>
                <Box sx={{ maxHeight: '300px', overflow: 'auto' }}>
                  {segments.map((segment, index) => (
                    <Paper key={index} sx={{ p: 2, mb: 1, bgcolor: '#fafafa' }}>
                      <Box display="flex" alignItems="center" gap={1} mb={1}>
                        <Chip
                          label={segment.speaker_id || 'Unknown'}
                          size="small"
                          color={segment.speaker_id === 'SPEAKER_0' ? 'primary' : 'secondary'}
                        />
                        <Typography variant="caption" color="textSecondary">
                          {segment.start_time?.toFixed(2)}s - {segment.end_time?.toFixed(2)}s
                        </Typography>
                        {segment.confidence > 0 && (
                          <Chip
                            label={`${(segment.confidence * 100).toFixed(1)}%`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                      <Typography variant="body2">
                        {segment.text || <em>No speech detected</em>}
                      </Typography>
                    </Paper>
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        )}

        {/* Tab Panel 1: Clinical Transcript */}
        {tabValue === 1 && clinicalData && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Clinical Transcript with Behavioral Analysis
            </Typography>

            {/* Session Summary */}
            {clinicalData.summary && (
              <Card sx={{ mb: 3, bgcolor: '#f0f7ff' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Session Summary
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6} md={3}>
                      <Typography variant="body2" color="textSecondary">Duration</Typography>
                      <Typography variant="h6">{clinicalData.summary.total_duration?.toFixed(1)}s</Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="body2" color="textSecondary">Total Words</Typography>
                      <Typography variant="h6">{clinicalData.summary.total_words}</Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="body2" color="textSecondary">Avg Response Latency</Typography>
                      <Typography variant="h6">{clinicalData.summary.average_response_latency?.toFixed(2)}s</Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="body2" color="textSecondary">Turn Count</Typography>
                      <Typography variant="h6">{clinicalData.summary.turn_count}</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}

            {/* Behavioral Patterns */}
            {clinicalData.behavioral_patterns && (
              <Card sx={{ mb: 3, bgcolor: '#fff8e1' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Behavioral Patterns Detected
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6} md={3}>
                      <Chip label={`Echolalia: ${clinicalData.behavioral_patterns.echolalia_count}`} color="info" />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Chip label={`Disfluency: ${clinicalData.behavioral_patterns.disfluency_count}`} color="warning" />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Chip label={`Elevated Latency: ${clinicalData.behavioral_patterns.elevated_latency_count}`} color="error" />
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Chip 
                        label={`Perseveration: ${clinicalData.behavioral_patterns.perseveration_detected ? 'Yes' : 'No'}`} 
                        color={clinicalData.behavioral_patterns.perseveration_detected ? 'error' : 'success'}
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}

            {/* Clinical Insights */}
            {clinicalData.clinical_insights && (
              <Card sx={{ mb: 3, bgcolor: '#e8f5e9' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Clinical Insights
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" color="textSecondary">Engagement Level</Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {clinicalData.clinical_insights.engagement_level}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" color="textSecondary">Communication Effectiveness</Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {clinicalData.clinical_insights.communication_effectiveness}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" color="textSecondary" gutterBottom>Strengths</Typography>
                      <List dense>
                        {clinicalData.clinical_insights.strengths?.map((strength, idx) => (
                          <ListItem key={idx}>
                            <ListItemText primary={`✓ ${strength}`} />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" color="textSecondary" gutterBottom>Areas of Concern</Typography>
                      <List dense>
                        {clinicalData.clinical_insights.areas_of_concern?.map((concern, idx) => (
                          <ListItem key={idx}>
                            <ListItemText primary={`⚠ ${concern}`} />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="body2" color="textSecondary" gutterBottom>Recommendations</Typography>
                      <List dense>
                        {clinicalData.clinical_insights.recommendations?.map((rec, idx) => (
                          <ListItem key={idx}>
                            <ListItemText primary={`→ ${rec}`} />
                          </ListItem>
                        ))}
                      </List>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}

            {/* Verbatim Transcript with Annotations */}
            <Typography variant="h6" gutterBottom>
              Verbatim Transcript
            </Typography>
            <Box sx={{ maxHeight: '500px', overflow: 'auto' }}>
              {clinicalData.segments?.map((segment, index) => (
                <Paper key={index} sx={{ p: 2, mb: 2, bgcolor: segment.speaker === 'therapist' ? '#e3f2fd' : '#fff3e0' }}>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Chip
                      label={segment.speaker?.toUpperCase()}
                      size="small"
                      color={segment.speaker === 'therapist' ? 'primary' : 'secondary'}
                    />
                    <Typography variant="caption" color="textSecondary">
                      {segment.start_time?.toFixed(2)}s - {segment.end_time?.toFixed(2)}s
                    </Typography>
                    {segment.sentiment && (
                      <Chip label={segment.sentiment} size="small" variant="outlined" />
                    )}
                    {segment.tone && (
                      <Chip label={segment.tone} size="small" variant="outlined" />
                    )}
                  </Box>
                  
                  <Typography variant="body1" sx={{ mb: 1, fontFamily: 'monospace' }}>
                    {segment.text}
                  </Typography>

                  {segment.behavioral_tags && segment.behavioral_tags.length > 0 && (
                    <Box display="flex" gap={0.5} mb={1}>
                      {segment.behavioral_tags.map((tag, idx) => (
                        <Chip key={idx} label={tag} size="small" color="warning" />
                      ))}
                    </Box>
                  )}

                  {segment.response_latency !== null && segment.response_latency !== undefined && (
                    <Typography variant="caption" color="textSecondary">
                      Response Latency: {segment.response_latency.toFixed(2)}s
                    </Typography>
                  )}

                  {segment.annotations && segment.annotations.length > 0 && (
                    <Box sx={{ mt: 1, pl: 2, borderLeft: '3px solid #ff9800' }}>
                      <Typography variant="caption" fontWeight="bold">Annotations:</Typography>
                      {segment.annotations.map((ann, idx) => (
                        <Box key={idx} sx={{ mt: 0.5 }}>
                          <Chip
                            label={ann.type}
                            size="small"
                            color={getSeverityColor(ann.severity)}
                            sx={{ mr: 1 }}
                          />
                          <Typography variant="caption">
                            {ann.description}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  )}
                </Paper>
              ))}
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
}

export default TranscriptViewer;
