import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  CircularProgress,
  Paper,
  Grid,
  Button,
  Tabs,
  Tab
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { fetchSession, fetchSessionFrames } from '../api/auditAPI';
import ScoreCharts from './ScoreCharts';
import FrameViewer from './FrameViewer';
import TranscriptViewer from './TranscriptViewer';
import FacialActionUnitsViewer from './FacialActionUnitsViewer';

function SessionDetail() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [frames, setFrames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    loadSessionData();
  }, [sessionId]);

  const loadSessionData = async () => {
    try {
      setLoading(true);
      const [sessionData, framesData] = await Promise.all([
        fetchSession(sessionId),
        fetchSessionFrames(sessionId)
      ]);
      setSession(sessionData);
      setFrames(framesData);
    } catch (error) {
      console.error('Failed to load session:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleMetricClick = (metricName) => {
    navigate(`/session/${sessionId}/metric/${metricName}`);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!session) {
    return (
      <Box textAlign="center" mt={4}>
        <Typography variant="h6" color="error">
          Session not found
        </Typography>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/')}>
          Back to Sessions
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/')}
        sx={{ mb: 2 }}
      >
        Back to Sessions
      </Button>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Session Details
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body1">
              <strong>Video:</strong> {session.video_path ? session.video_path.split(/[\\/]/).pop() : 'Unknown'}
            </Typography>
            <Typography variant="body1">
              <strong>Date:</strong> {new Date(session.timestamp).toLocaleString()}
            </Typography>
            <Typography variant="body1">
              <strong>Duration:</strong> {(session.duration || 0).toFixed(2)}s
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body1">
              <strong>Total Frames:</strong> {session.total_frames}
            </Typography>
            <Typography variant="body1">
              <strong>Faces Detected:</strong> {session.frame_count_face || 0}
            </Typography>
            <Typography variant="body1">
              <strong>Poses Detected:</strong> {session.frame_count_pose || 0}
            </Typography>
          </Grid>
        </Grid>

        {(session.vocal_regulation_index !== null || session.motor_agitation_index !== null) && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Behavioral Scores
            </Typography>
            <Grid container spacing={2}>
              {session.vocal_regulation_index !== null && (
                <Grid item xs={12} md={3}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'primary.light',
                      color: 'primary.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('vocal_regulation')}
                  >
                    <Typography variant="h6">
                      {session.vocal_regulation_index.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Vocal Regulation (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
              {session.motor_agitation_index !== null && (
                <Grid item xs={12} md={3}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'secondary.light',
                      color: 'secondary.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('motor_agitation')}
                  >
                    <Typography variant="h6">
                      {session.motor_agitation_index.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Motor Agitation (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
              {session.attention_stability_score !== null && (
                <Grid item xs={12} md={3}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'info.light',
                      color: 'info.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('attention_stability')}
                  >
                    <Typography variant="h6">
                      {session.attention_stability_score.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Attention Stability (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
              {/* Regulation Consistency Card Removed */}
              {session.facial_affect_index !== null && (
                <Grid item xs={12} md={3}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'success.light',
                      color: 'success.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => setActiveTab(2)} // Switch to Facial Action Units tab
                  >
                    <Typography variant="h6">
                      {session.facial_affect_index.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Facial Affect Index (Click for AU details)</Typography>
                  </Paper>
                </Grid>
              )}
            </Grid>
          </Box>
        )}

        {(session.turn_taking_score !== null || session.eye_contact_score !== null || session.social_engagement_index !== null) && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Autism-Specific Analysis
            </Typography>
            <Grid container spacing={2}>
              {session.turn_taking_score !== null && (
                <Grid item xs={12} md={4}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'success.light',
                      color: 'success.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('turn_taking')}
                  >
                    <Typography variant="h6">
                      {session.turn_taking_score.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Turn-Taking (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
              {session.eye_contact_score !== null && (
                <Grid item xs={12} md={4}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'info.light',
                      color: 'info.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('eye_contact')}
                  >
                    <Typography variant="h6">
                      {session.eye_contact_score.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Eye Contact (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
              {/* Social Engagement Card Removed */}
            </Grid>
          </Box>
        )}

        {(session.stuttering_severity_index !== null || session.responsiveness_index !== null) && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Clinical Analysis
            </Typography>
            <Grid container spacing={2}>
              {session.stuttering_severity_index !== null && (
                <Grid item xs={12} md={6}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'warning.light',
                      color: 'warning.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('stuttering')}
                  >
                    <Typography variant="h6">
                      {session.stuttering_severity_index.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Stuttering Severity (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
              {session.responsiveness_index !== null && (
                <Grid item xs={12} md={6}>
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: 'success.light',
                      color: 'success.contrastText',
                      cursor: 'pointer',
                      '&:hover': { boxShadow: 6, transform: 'scale(1.02)' },
                      transition: 'all 0.2s'
                    }}
                    onClick={() => handleMetricClick('responsiveness')}
                  >
                    <Typography variant="h6">
                      {session.responsiveness_index.toFixed(1)}/100
                    </Typography>
                    <Typography variant="body2">Responsiveness (Click for audit)</Typography>
                  </Paper>
                </Grid>
              )}
            </Grid>
          </Box>
        )}
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
          {/* <Tab label="Score Charts" />
          <Tab label="Frame Analysis" /> */}
          <Tab label="Facial Action Units" />
          <Tab label="Transcript" />
        </Tabs>

        {/* {activeTab === -2 && <ScoreCharts frames={frames} />}
        {activeTab === -1 && <FrameViewer frames={frames} />} */}
        {activeTab === 0 && <FacialActionUnitsViewer sessionId={sessionId} />}
        {activeTab === 1 && <TranscriptViewer sessionId={sessionId} />}
      </Paper>
    </Box>
  );
}

export default SessionDetail;
