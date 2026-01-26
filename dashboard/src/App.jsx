import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  CircularProgress
} from '@mui/material';
import SessionList from './components/SessionList';
import SessionDetail from './components/SessionDetail';
import Statistics from './components/Statistics';
import VideoUpload from './components/VideoUpload';
import MetricAudit from './components/MetricAudit';

function App() {
  return (
    <Router>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Behavior Scope Dashboard
            </Typography>
            <Link to="/" style={{ color: 'white', marginRight: '20px', textDecoration: 'none' }}>
              Sessions
            </Link>
            <Link to="/upload" style={{ color: 'white', marginRight: '20px', textDecoration: 'none' }}>
              Upload Video
            </Link>
            <Link to="/statistics" style={{ color: 'white', textDecoration: 'none' }}>
              Statistics
            </Link>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4 }}>
          <Routes>
            <Route path="/" element={<SessionList />} />
            <Route path="/upload" element={<VideoUpload />} />
            <Route path="/session/:sessionId" element={<SessionDetail />} />
            <Route path="/session/:sessionId/metric/:metricName" element={<MetricAudit />} />
            <Route path="/statistics" element={<Statistics />} />
          </Routes>
        </Container>
      </Box>
    </Router>
  );
}

export default App;
