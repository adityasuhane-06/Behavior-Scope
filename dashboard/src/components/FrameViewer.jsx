import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Chip,
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

function FrameViewer({ frames }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  if (!frames || frames.length === 0) {
    return (
      <Box textAlign="center" py={4}>
        <Typography variant="h6" color="textSecondary">
          No frame data available
        </Typography>
      </Box>
    );
  }

  const paginatedFrames = frames.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  return (
    <Box>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Frame</strong></TableCell>
              <TableCell><strong>Time (s)</strong></TableCell>
              <TableCell><strong>Detections</strong></TableCell>
              <TableCell><strong>Engagement</strong></TableCell>
              <TableCell><strong>Autism Risk</strong></TableCell>
              <TableCell><strong>Eye Contact</strong></TableCell>
              <TableCell><strong>Movement</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedFrames.map((frame) => (
              <React.Fragment key={frame.frame_number}>
                <TableRow hover>
                  <TableCell>{frame.frame_number}</TableCell>
                  <TableCell>{parseFloat(frame.timestamp || 0).toFixed(2)}</TableCell>
                  <TableCell>
                    {frame.face_detected && (
                      <Chip label="Face" size="small" color="primary" sx={{ mr: 0.5 }} />
                    )}
                    {frame.pose_detected && (
                      <Chip label="Pose" size="small" color="secondary" sx={{ mr: 0.5 }} />
                    )}
                    {frame.eye_contact_detected && (
                      <Chip label="Eyes" size="small" color="success" sx={{ mr: 0.5 }} />
                    )}
                    {frame.movement_detected && (
                      <Chip label="Movement" size="small" color="warning" />
                    )}
                  </TableCell>
                  <TableCell>
                    {frame.confidence_score
                      ? `${(frame.confidence_score * 100).toFixed(1)}%`
                      : '-'}
                  </TableCell>
                  <TableCell>
                    {frame.analysis_type === 'autism_analysis' ? 'Analyzed' : '-'}
                  </TableCell>
                  <TableCell>
                    {frame.eye_contact_detected ? 'Detected' : '-'}
                  </TableCell>
                  <TableCell>
                    {frame.movement_detected ? 'Detected' : '-'}
                  </TableCell>
                </TableRow>

                {/* Expandable details for frames with detections */}
                {(frame.face_detected || frame.pose_detected || frame.eye_contact_detected || frame.movement_detected) && (
                  <TableRow>
                    <TableCell colSpan={7} sx={{ py: 0, bgcolor: '#f5f5f5' }}>
                      <Accordion elevation={0}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="body2">View Detection Details</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            {frame.face_detected && (
                              <Box>
                                <Typography variant="subtitle2" color="primary">
                                  Face Detection
                                </Typography>
                                <Typography variant="body2">
                                  Status: Detected
                                </Typography>
                                {frame.confidence_score && (
                                  <Typography variant="body2">
                                    Confidence: {(frame.confidence_score * 100).toFixed(1)}%
                                  </Typography>
                                )}
                              </Box>
                            )}

                            {frame.eye_contact_detected && (
                              <Box>
                                <Typography variant="subtitle2" color="success.main">
                                  Eye Contact
                                </Typography>
                                <Typography variant="body2">
                                  Status: Detected
                                </Typography>
                              </Box>
                            )}

                            {frame.movement_detected && (
                              <Box>
                                <Typography variant="subtitle2" color="warning.main">
                                  Movement
                                </Typography>
                                <Typography variant="body2">
                                  Status: Detected
                                </Typography>
                              </Box>
                            )}

                            {frame.pose_detected && (
                              <Box>
                                <Typography variant="subtitle2" color="secondary">
                                  Pose Detection
                                </Typography>
                                <Typography variant="body2">
                                  Status: Detected
                                </Typography>
                              </Box>
                            )}

                            {frame.action_units && frame.action_units !== '[]' && (
                              <Box>
                                <Typography variant="subtitle2" color="info.main">
                                  Action Units
                                </Typography>
                                <Typography variant="body2">
                                  Data: {frame.action_units}
                                </Typography>
                              </Box>
                            )}

                            {frame.details && frame.details !== '{}' && (
                              <Box>
                                <Typography variant="subtitle2" color="text.secondary">
                                  Additional Details
                                </Typography>
                                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                  {frame.details}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        </AccordionDetails>
                      </Accordion>
                    </TableCell>
                  </TableRow>
                )}
              </React.Fragment>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={frames.length}
        page={page}
        onPageChange={handleChangePage}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        rowsPerPageOptions={[10, 25, 50, 100]}
      />
    </Box>
  );
}

export default FrameViewer;
