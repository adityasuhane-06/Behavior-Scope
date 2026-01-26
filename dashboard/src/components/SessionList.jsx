import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Box,
  Chip,
  TextField,
  Button
} from '@mui/material';
import { fetchSessions, searchSessions } from '../api/auditAPI';

function SessionList() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchVideo, setSearchVideo] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const data = await fetchSessions();
      setSessions(data);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    try {
      setLoading(true);
      const data = await searchSessions({ video: searchVideo });
      setSessions(data);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSessionClick = (sessionId) => {
    navigate(`/session/${sessionId}`);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analysis Sessions
      </Typography>

      <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
        <TextField
          label="Search by video name"
          variant="outlined"
          value={searchVideo}
          onChange={(e) => setSearchVideo(e.target.value)}
          sx={{ flexGrow: 1 }}
        />
        <Button variant="contained" onClick={handleSearch}>
          Search
        </Button>
        <Button variant="outlined" onClick={loadSessions}>
          Clear
        </Button>
      </Box>

      <Grid container spacing={3}>
        {sessions.map((session) => (
          <Grid item xs={12} md={6} lg={4} key={session.session_id}>
            <Card
              sx={{ cursor: 'pointer', '&:hover': { boxShadow: 6 } }}
              onClick={() => handleSessionClick(session.session_id)}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom noWrap>
                  {session.video_path ? session.video_path.split(/[\\/]/).pop() : 'Unknown'}
                </Typography>
                <Typography color="textSecondary" gutterBottom>
                  {new Date(session.timestamp).toLocaleString()}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={`${session.total_frames} frames`}
                    size="small"
                    sx={{ mr: 1, mb: 1 }}
                  />
                  <Chip
                    label={`${session.frame_count_face || 0} faces`}
                    size="small"
                    color="primary"
                    sx={{ mr: 1, mb: 1 }}
                  />
                  <Chip
                    label={`${session.frame_count_pose || 0} poses`}
                    size="small"
                    color="secondary"
                    sx={{ mr: 1, mb: 1 }}
                  />
                </Box>
                {session.overall_scores && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      Engagement: {(session.overall_scores.engagement_score * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">
                      Autism Risk: {(session.overall_scores.autism_risk_score * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {sessions.length === 0 && (
        <Box textAlign="center" mt={4}>
          <Typography variant="h6" color="textSecondary">
            No sessions found
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default SessionList;
