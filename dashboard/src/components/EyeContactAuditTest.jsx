import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Paper
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

function EyeContactAuditTest() {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  return (
    <Box>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(`/session/${sessionId}`)}
        sx={{ mb: 2 }}
      >
        Back to Session
      </Button>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Eye Contact Audit Test
        </Typography>
        <Typography variant="h6" color="primary" gutterBottom>
          Session: {sessionId}
        </Typography>
        <Typography variant="body1">
          This is a test version of the Eye Contact Audit component.
          If you can see this, the routing and basic component loading is working.
        </Typography>
      </Paper>
    </Box>
  );
}

export default EyeContactAuditTest;