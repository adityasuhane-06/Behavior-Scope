import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Divider,
  CircularProgress,
  Alert,
  Tooltip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VisibilityIcon from '@mui/icons-material/Visibility';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import axios from 'axios';

const AU_DESCRIPTIONS = {
  1: {
    name: "Inner Brow Raiser",
    description: "Eyebrows raised (inner) - Often seen during concentration, worry, or surprise",
    muscle: "Frontalis, pars medialis",
    clinical: "Associated with surprise, concern, sadness"
  },
  2: {
    name: "Outer Brow Raiser", 
    description: "Eyebrows raised (outer) - Often seen during surprise, fear",
    muscle: "Frontalis, pars lateralis",
    clinical: "Associated with surprise, fear"
  },
  4: {
    name: "Brow Lowerer",
    description: "Eyebrows furrowed - Often seen during concentration, anger, confusion",
    muscle: "Corrugator supercilii, depressor supercilii",
    clinical: "Associated with concentration, anger, confusion"
  },
  5: {
    name: "Upper Lid Raiser",
    description: "Eyes widened - Often seen during surprise, fear, wide-eyed attention",
    muscle: "Levator palpebrae superioris",
    clinical: "Associated with surprise, fear, wide-eyed attention"
  },
  6: {
    name: "Cheek Raiser",
    description: "Cheeks raised - Genuine smile indicator (Duchenne smile)",
    muscle: "Orbicularis oculi, pars orbitalis",
    clinical: "Genuine smile indicator (Duchenne smile)"
  },
  7: {
    name: "Lid Tightener",
    description: "Eyes tightened - Often seen during concentration, squinting, anger",
    muscle: "Orbicularis oculi, pars palpebralis",
    clinical: "Associated with concentration, squinting, anger"
  },
  9: {
    name: "Nose Wrinkler",
    description: "Nose wrinkled - Often seen during disgust",
    muscle: "Levator labii superioris alaeque nasi",
    clinical: "Associated with disgust"
  },
  10: {
    name: "Upper Lip Raiser",
    description: "Upper lip raised - Often seen during disgust, contempt, smirking",
    muscle: "Levator labii superioris",
    clinical: "Associated with disgust, contempt, smirking"
  },
  12: {
    name: "Lip Corner Puller",
    description: "Mouth corners pulled up - PRIMARY smile AU",
    muscle: "Zygomatic major",
    clinical: "Smile indicator (both genuine and social smiles)"
  },
  15: {
    name: "Lip Corner Depressor",
    description: "Mouth corners pulled down - Opposite of smile",
    muscle: "Depressor anguli oris",
    clinical: "Associated with sadness, frowning"
  },
  17: {
    name: "Chin Raiser",
    description: "Chin raised - Often seen during doubt, sadness, 'pouty' expression",
    muscle: "Mentalis",
    clinical: "Associated with doubt, sadness, 'pouty' expression"
  },
  20: {
    name: "Lip Stretcher",
    description: "Lips stretched horizontally - Often seen during fear, tension",
    muscle: "Risorius",
    clinical: "Associated with fear, tension"
  },
  23: {
    name: "Lip Tightener",
    description: "Lips tightened - Often seen during anger, concentration, suppressing expression",
    muscle: "Orbicularis oris",
    clinical: "Associated with anger, concentration, suppressing expression"
  },
  25: {
    name: "Lips Part",
    description: "Lips parted - Relaxed state, speaking, surprise",
    muscle: "Relaxation of mentalis or lip depressor",
    clinical: "Relaxed state, speaking, surprise"
  },
  26: {
    name: "Jaw Drop",
    description: "Jaw dropped/mouth open - Surprise, shock, mouth opening for speech",
    muscle: "Masseter relaxation, lateral pterygoid",
    clinical: "Surprise, shock, mouth opening for speech"
  }
};

// Helper component to show all 15 AUs in a tooltip
const All15AUsTooltip = () => {
  return (
    <Box sx={{ maxWidth: 400 }}>
      <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
        What are the 15 Action Units?
      </Typography>
      <Typography variant="body2" sx={{ mb: 1, fontSize: '0.85rem' }}>
        These are the 15 specific facial muscle movements that the system can detect and track:
      </Typography>
      <Box sx={{ fontSize: '0.8rem', whiteSpace: 'pre-line' }}>
        {Object.entries(AU_DESCRIPTIONS)
          .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
          .map(([num, info]) => (
            <Box key={num} sx={{ mb: 0.5 }}>
              <strong>AU{num}:</strong> {info.name}
            </Box>
          ))}
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic', color: 'text.secondary' }}>
        This is the standard FACS (Facial Action Coding System) subset used in clinical research.
      </Typography>
    </Box>
  );
};

// Helper component to show detected AUs in a tooltip
const DetectedAUsTooltip = ({ detectedAUs }) => {
  if (!detectedAUs || detectedAUs.length === 0) {
    return <Typography variant="body2">No AUs detected</Typography>;
  }
  
  return (
    <Box sx={{ maxWidth: 400 }}>
      <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
        What are the {detectedAUs.length} Detected AUs?
      </Typography>
      <Typography variant="body2" sx={{ mb: 1, fontSize: '0.85rem' }}>
        These are the specific facial movements that were actually observed in this video:
      </Typography>
      <Box sx={{ fontSize: '0.8rem' }}>
        {detectedAUs.map((auNum) => (
          <Box key={auNum} sx={{ mb: 0.5 }}>
            <strong>AU{auNum}:</strong> {AU_DESCRIPTIONS[auNum]?.name || 'Unknown'}
          </Box>
        ))}
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic', color: 'text.secondary' }}>
        Out of 15 possible AUs, {detectedAUs.length} were detected = {((detectedAUs.length / 15) * 100).toFixed(1)}% variety
      </Typography>
    </Box>
  );
};

// Helper component to explain symmetry calculation
const SymmetryExplanationTooltip = ({ symmetryData }) => {
  // If we have actual calculation data from backend, show it
  if (symmetryData && symmetryData.inputs) {
    const symmetryScore = symmetryData.result || symmetryData.inputs.symmetry_score || 0;
    const asymmetry = symmetryData.inputs.asymmetry_percentage || 0;
    const framesAnalyzed = symmetryData.inputs.total_frames_with_face || 0;
    
    return (
      <Box sx={{ maxWidth: 500 }}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="primary">
          Facial Symmetry Calculation
        </Typography>
        
        <Paper sx={{ p: 1.5, mb: 1.5, bgcolor: 'success.light' }}>
          <Typography variant="body2" fontWeight="bold" gutterBottom>
            📊 Result:
          </Typography>
          <Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', fontWeight: 'bold' }}>
            Symmetry Score = {symmetryScore.toFixed(2)}/100
          </Box>
          <Typography variant="caption" color="text.secondary">
            (Asymmetry: {asymmetry.toFixed(2)}%)
          </Typography>
        </Paper>
        
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Frames Analyzed:</strong> {framesAnalyzed}
        </Typography>
        
        <Typography variant="body2" sx={{ mb: 0.5, fontWeight: 'bold' }}>
          Landmark Pairs Compared:
        </Typography>
        <Box sx={{ fontSize: '0.8rem', ml: 1, mb: 1.5 }}>
          {symmetryData.inputs.landmark_pairs?.map((pair, idx) => (
            <div key={idx}>• {pair}</div>
          ))}
        </Box>
        
        <Paper sx={{ p: 1, bgcolor: 'info.light' }}>
          <Typography variant="caption" sx={{ fontSize: '0.75rem', display: 'block' }}>
            <strong>Method:</strong> {symmetryData.inputs.calculation_method || 'Geometric landmark comparison'}
          </Typography>
          <Typography variant="caption" sx={{ fontSize: '0.7rem', display: 'block', mt: 0.5, fontStyle: 'italic' }}>
            {symmetryData.note || 'Compares left and right facial landmarks to measure balance'}
          </Typography>
        </Paper>
      </Box>
    );
  }
  
  // Fallback if no data
  return (
    <Box sx={{ maxWidth: 400 }}>
      <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
        Facial Symmetry
      </Typography>
      <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
        Measures left-right balance by comparing 5 pairs of facial landmarks (eyes, mouth, cheeks, eyebrows, jaw).
      </Typography>
    </Box>
  );
};

function FacialActionUnitsViewer({ sessionId }) {
  const [auData, setAuData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedAU, setSelectedAU] = useState(null);
  const [auDetailsDialog, setAuDetailsDialog] = useState(false);
  const [frameEvidence, setFrameEvidence] = useState([]);
  const [loadingEvidence, setLoadingEvidence] = useState(false);
  const [formulaDialog, setFormulaDialog] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [videoPlayerRef, setVideoPlayerRef] = useState(null);

  useEffect(() => {
    if (sessionId) {
      loadFacialActionUnits();
    }
  }, [sessionId]);

  const handleTimestampClick = (timestamp) => {
    if (videoPlayerRef) {
      videoPlayerRef.currentTime = timestamp;
      videoPlayerRef.pause();
    }
  };

  const loadFacialActionUnits = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        `http://localhost:8000/sessions/${sessionId}/facial-action-units`
      );
      setAuData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load facial action units data');
    } finally {
      setLoading(false);
    }
  };

  const handleAUClick = async (auNumber) => {
    setSelectedAU(auNumber);
    setLoadingEvidence(true);
    setAuDetailsDialog(true);

    try {
      const response = await axios.get(
        `http://localhost:8000/sessions/${sessionId}/facial-action-units/${auNumber}/evidence`
      );
      setFrameEvidence(response.data.evidence || []);
    } catch (err) {
      console.error('Failed to load AU evidence:', err);
      setFrameEvidence([]);
    } finally {
      setLoadingEvidence(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, '0')}`;
  };

  const getInputExplanation = (key, value) => {
    const explanations = {
      'unique_aus': (() => {
        // Get detected AUs from the data
        const detectedAUs = auData?.calculation_details?.affect_range?.inputs?.detected_aus || [];
        const detectedCount = detectedAUs.length;
        const totalPossible = 15;
        const notDetected = Object.keys(AU_DESCRIPTIONS)
          .map(num => parseInt(num))
          .filter(num => !detectedAUs.includes(num));
        
        return (
          <Box>
            <Typography variant="body2" sx={{ mb: 1 }}>
              This is the COUNT of how many DIFFERENT Action Units were activated.
            </Typography>
            <Typography variant="body2" fontWeight="bold" sx={{ mb: 1 }}>
              In this video: {detectedCount} detected out of {totalPossible} possible
            </Typography>
            
            <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 1.5, mb: 0.5, color: 'success.main' }}>
              ✓ {detectedCount} Detected:
            </Typography>
            <Box sx={{ ml: 2, fontSize: '0.85rem' }}>
              {detectedAUs.map(au => (
                <div key={au}>• AU{au}: {AU_DESCRIPTIONS[au]?.name}</div>
              ))}
            </Box>
            
            <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 1.5, mb: 0.5, color: 'error.main' }}>
              ✗ {notDetected.length} NOT Detected:
            </Typography>
            <Box sx={{ ml: 2, fontSize: '0.85rem', color: 'text.secondary' }}>
              {notDetected.map(au => (
                <div key={au}>• AU{au}: {AU_DESCRIPTIONS[au]?.name}</div>
              ))}
            </Box>
          </Box>
        );
      })(),
      'max_possible_aus': `This is the TOTAL number of Action Units the system can track. The system monitors 15 specific facial muscle movements based on the FACS (Facial Action Coding System).`,
      'detected_aus': `These are the SPECIFIC AU numbers that were found: ${Array.isArray(value) ? value.map(au => `AU${au} (${AU_DESCRIPTIONS[au]?.name || 'Unknown'})`).join(', ') : value}`,
      'avg_intensity': `Average intensity of all AU activations across all frames. Range: 0.0 (no activation) to 1.0 (maximum activation). Measured from facial landmark displacement.`,
      'avg_activations_per_frame': `Average number of AUs activated per frame. Higher values indicate more simultaneous facial movements.`,
      'intensity_score': `Normalized intensity score (0-100). Calculated by multiplying average intensity by 100.`,
      'activation_score': `Normalized activation frequency score (0-100). Based on how many AUs are active per frame compared to expected baseline of 3 AUs.`,
      'diversity_score': `Ratio of unique AUs to maximum possible AUs. Range: 0.0 (no diversity) to 1.0 (all AUs activated).`,
      'low_diversity': `Inverse of diversity, weighted by 40%. Higher values indicate limited facial expression variety.`,
      'low_intensity': `Inverse of intensity, weighted by 35%. Higher values indicate weak facial movements.`,
      'inactive_frames': `Number of frames where no Action Units were activated (completely still face).`,
      'total_frames': `Total number of video frames analyzed in this session.`,
      'inactive_ratio': `Proportion of frames with no AU activations. Range: 0.0 (always active) to 1.0 (never active).`,
      'inactive_score': `Contribution of inactive frames to flat affect score, weighted by 25%.`,
      'affect_range': `Final affect range score (0-100). Measures diversity of facial expressions.`,
      'mobility': `Final mobility score (0-100). Measures amount and intensity of facial movement.`,
      'flat_affect': `Final flat affect score (0-100). Higher scores indicate reduced expressiveness.`,
      'flat_affect_inverted': `Inverted flat affect score (100 - flat_affect). Used in composite calculation so higher is better.`,
      'congruence': `Context appropriateness score (0-100). Measures if facial patterns match expected context.`,
      'symmetry': `Left-right facial balance score (0-100). Measures bilateral symmetry of facial movements.`,
      'method': value
    };
    
    return explanations[key] || `Input parameter: ${key}`;
  };

  const getIntensityColor = (intensity) => {
    if (intensity >= 0.7) return '#d32f2f'; // High - Red
    if (intensity >= 0.5) return '#f57c00'; // Medium - Orange  
    if (intensity >= 0.3) return '#fbc02d'; // Low-Medium - Yellow
    return '#388e3c'; // Low - Green
  };

  const getIntensityLabel = (intensity) => {
    if (intensity >= 0.7) return 'High';
    if (intensity >= 0.5) return 'Medium';
    if (intensity >= 0.3) return 'Low-Medium';
    return 'Low';
  };

  const handleMetricClick = (metricName) => {
    setSelectedMetric(metricName);
    setFormulaDialog(true);
  };

  const getMetricFormula = (metricName) => {
    // If we have calculation_details from API, use that
    const calcDetails = auData?.calculation_details;
    
    if (calcDetails && calcDetails[metricName]) {
      const detail = calcDetails[metricName];
      return {
        name: metricName.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
        formula: detail.formula,
        components: Object.entries(detail.inputs || {}).map(([key, value]) => ({
          name: key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
          value: typeof value === 'number' ? value.toFixed(2) : JSON.stringify(value),
          description: `Input value for ${key}`,
          calculation: `${key} = ${typeof value === 'number' ? value.toFixed(2) : JSON.stringify(value)}`
        })),
        finalCalculation: detail.step_by_step ? detail.step_by_step.join('\n') : detail.calculation,
        result: detail.result,
        interpretation: detail.interpretation
      };
    }
    
    // Fallback to static formulas if API doesn't provide details
    const formulas = {
      'facial_affect_index': {
        name: 'Facial Affect Index (FAI)',
        formula: 'FAI = (0.3 × Affect Range) + (0.3 × Mobility) + (0.2 × (100 - Flat Affect)) + (0.2 × Symmetry)',
        components: [
          {
            name: 'Affect Range',
            value: auData?.affect_range?.toFixed(2),
            description: 'Variety of different Action Units activated',
            calculation: `Number of unique AUs / Total possible AUs × 100 = ${auData?.most_activated_units?.length || 0} / 15 × 100 = ${auData?.affect_range?.toFixed(2)}`
          },
          {
            name: 'Mobility',
            value: auData?.mobility?.toFixed(2),
            description: 'Average intensity of AU activations',
            calculation: `Average of all AU intensities × 100 = ${auData?.mobility?.toFixed(2)}`
          },
          {
            name: 'Flat Affect (inverted)',
            value: auData?.flat_affect?.toFixed(2),
            description: 'Inverse of facial expressiveness (lower is better)',
            calculation: `100 - Flat Affect Score = 100 - ${auData?.flat_affect?.toFixed(2)} = ${(100 - (auData?.flat_affect || 0)).toFixed(2)}`
          },
          {
            name: 'Symmetry',
            value: auData?.symmetry?.toFixed(2),
            description: 'Balance between left and right facial movements',
            calculation: `Symmetry score = ${auData?.symmetry?.toFixed(2)}`
          }
        ],
        finalCalculation: `FAI = (0.3 × ${auData?.affect_range?.toFixed(2)}) + (0.3 × ${auData?.mobility?.toFixed(2)}) + (0.2 × ${(100 - (auData?.flat_affect || 0)).toFixed(2)}) + (0.2 × ${auData?.symmetry?.toFixed(2)})
     = ${(0.3 * (auData?.affect_range || 0)).toFixed(2)} + ${(0.3 * (auData?.mobility || 0)).toFixed(2)} + ${(0.2 * (100 - (auData?.flat_affect || 0))).toFixed(2)} + ${(0.2 * (auData?.symmetry || 0)).toFixed(2)}
     = ${auData?.facial_affect_index?.toFixed(2)}/100`,
        interpretation: auData?.facial_affect_index >= 70 
          ? 'High expressiveness - Wide range of facial movements with good mobility'
          : auData?.facial_affect_index >= 50
          ? 'Moderate expressiveness - Adequate facial movement range'
          : 'Low expressiveness - Limited facial movement variety or intensity'
      },
      'affect_range': {
        name: 'Affect Range',
        formula: 'Affect Range = (Number of Unique AUs Activated / Total Possible AUs) × 100',
        components: [
          {
            name: 'Unique AUs Detected',
            value: auData?.most_activated_units?.length || 0,
            description: 'Number of different Action Units that were activated during the session',
            calculation: `Detected AUs: ${auData?.most_activated_units?.map(au => `AU${au.au_number}`).join(', ')}`
          },
          {
            name: 'Total Possible AUs',
            value: 15,
            description: 'Total number of Action Units tracked by the system',
            calculation: 'Standard FACS subset: 15 AUs'
          }
        ],
        finalCalculation: `Affect Range = (${auData?.most_activated_units?.length || 0} / 15) × 100 = ${auData?.affect_range?.toFixed(2)}`,
        interpretation: auData?.affect_range >= 80
          ? 'Wide variety - Highly expressive with diverse facial movements'
          : auData?.affect_range >= 50
          ? 'Moderate variety - Good range of facial expressions'
          : 'Limited variety - Restricted range of facial movements'
      },
      'mobility': {
        name: 'Facial Mobility',
        formula: 'Mobility = Average Intensity of All AU Activations × 100',
        components: [
          {
            name: 'AU Intensities',
            value: auData?.most_activated_units?.map(au => `AU${au.au_number}: ${(au.avg_intensity * 100).toFixed(1)}%`).join(', '),
            description: 'Average intensity for each activated Action Unit',
            calculation: 'Measured from facial landmark movements'
          },
          {
            name: 'Overall Average',
            value: auData?.mobility?.toFixed(2),
            description: 'Mean of all AU intensities',
            calculation: `Sum of intensities / Number of AUs = ${auData?.mobility?.toFixed(2)}`
          }
        ],
        finalCalculation: `Mobility = ${auData?.mobility?.toFixed(2)}/100`,
        interpretation: auData?.mobility >= 70
          ? 'High mobility - Strong, pronounced facial movements'
          : auData?.mobility >= 50
          ? 'Moderate mobility - Normal facial movement intensity'
          : 'Low mobility - Subtle or restricted facial movements'
      },
      'symmetry': {
        name: 'Facial Symmetry',
        formula: 'Symmetry = 100 - |Left Side Activation - Right Side Activation|',
        components: [
          {
            name: 'Left-Right Balance',
            value: auData?.symmetry?.toFixed(2),
            description: 'Comparison of facial movements on left vs right side',
            calculation: 'Measured from bilateral AU activations'
          }
        ],
        finalCalculation: `Symmetry = ${auData?.symmetry?.toFixed(2)}/100`,
        interpretation: auData?.symmetry >= 80
          ? 'Highly symmetric - Balanced facial movements'
          : auData?.symmetry >= 60
          ? 'Moderately symmetric - Some asymmetry present'
          : 'Asymmetric - Notable differences between left and right sides'
      }
    };
    
    return formulas[metricName] || null;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
        <Typography variant="body1" sx={{ ml: 2 }}>
          Loading Facial Action Units...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!auData || !auData.most_activated_units) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        No facial action units data available for this session.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <PhotoCameraIcon sx={{ mr: 1 }} />
        Facial Action Units Analysis (FACS)
      </Typography>
      
      <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
        Objective measurement of facial muscle activations based on the Facial Action Coding System (FACS). 
        This analysis describes <strong>what facial movements occurred</strong>, not emotions.
        Click on any AU to see detailed time frames and photo evidence.
      </Typography>

      {/* Overall Statistics */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={3}>
            <Card 
              sx={{ 
                bgcolor: 'primary.light', 
                color: 'primary.contrastText',
                cursor: 'pointer',
                '&:hover': { 
                  boxShadow: 6, 
                  transform: 'scale(1.02)',
                },
                transition: 'all 0.2s'
              }}
              onClick={() => handleMetricClick('facial_affect_index')}
            >
              <CardContent>
                <Typography variant="h4">
                  {auData.facial_affect_index?.toFixed(1) || 'N/A'}/100
                </Typography>
                <Typography variant="body2">Facial Affect Index</Typography>
                <Typography variant="caption">
                  Composite score reflecting facial expressiveness, mobility, and affect range
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                  Click to see formula
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card
              sx={{ 
                cursor: 'pointer',
                '&:hover': { 
                  boxShadow: 6, 
                  transform: 'scale(1.02)',
                },
                transition: 'all 0.2s'
              }}
              onClick={() => handleMetricClick('affect_range')}
            >
              <CardContent>
                <Typography variant="h4" color="primary">
                  {auData.affect_range?.toFixed(1) || 'N/A'}
                </Typography>
                <Typography variant="body2">Affect Range</Typography>
                <Typography variant="caption">Wide variety</Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                  Click to see formula
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card
              sx={{ 
                cursor: 'pointer',
                '&:hover': { 
                  boxShadow: 6, 
                  transform: 'scale(1.02)',
                },
                transition: 'all 0.2s'
              }}
              onClick={() => handleMetricClick('mobility')}
            >
              <CardContent>
                <Typography variant="h4" color="secondary">
                  {auData.mobility?.toFixed(1) || 'N/A'}
                </Typography>
                <Typography variant="body2">Mobility</Typography>
                <Typography variant="caption">Moderate movement</Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                  Click to see formula
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card
              sx={{ 
                cursor: 'pointer',
                '&:hover': { 
                  boxShadow: 6, 
                  transform: 'scale(1.02)',
                },
                transition: 'all 0.2s'
              }}
              onClick={() => handleMetricClick('symmetry')}
            >
              <CardContent>
                <Typography variant="h4" color="info.main">
                  {auData.symmetry?.toFixed(1) || 'N/A'}
                </Typography>
                <Typography variant="body2">Symmetry</Typography>
                <Typography variant="caption">Asymmetric</Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                  Click to see formula
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* Most Frequently Activated Action Units */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <VisibilityIcon sx={{ mr: 1 }} />
          Most Frequently Activated Action Units
        </Typography>
        
        <Grid container spacing={2}>
          {auData.most_activated_units.map((au) => (
            <Grid item xs={12} md={6} key={au.au_number}>
              <Card 
                sx={{ 
                  cursor: 'pointer',
                  '&:hover': { 
                    boxShadow: 6, 
                    transform: 'scale(1.02)',
                    bgcolor: 'action.hover'
                  },
                  transition: 'all 0.2s',
                  border: `2px solid ${getIntensityColor(au.max_intensity)}`
                }}
                onClick={() => handleAUClick(au.au_number)}
              >
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                    <Box flex={1}>
                      <Typography variant="h6" color="primary">
                        AU{au.au_number}: {AU_DESCRIPTIONS[au.au_number]?.name || 'Unknown AU'}
                      </Typography>
                      <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                        {AU_DESCRIPTIONS[au.au_number]?.description || 'No description available'}
                      </Typography>
                      <Box display="flex" gap={1} flexWrap="wrap">
                        <Tooltip 
                          title={
                            <Box sx={{ maxWidth: 450 }}>
                              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                Max Intensity: {(au.max_intensity * 100).toFixed(1)}%
                              </Typography>
                              <Divider sx={{ my: 1 }} />
                              <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>
                                <strong>How intensity is calculated:</strong>
                              </Typography>
                              <Box sx={{ fontFamily: 'monospace', fontSize: '0.75rem', bgcolor: 'rgba(0,0,0,0.1)', p: 1, borderRadius: 1, mb: 1 }}>
                                1. Measure landmark displacement (pixels)<br/>
                                2. Normalize by face size<br/>
                                3. Compare to neutral baseline<br/>
                                4. intensity = (measured - baseline) / range<br/>
                                5. Clip to 0-100%
                              </Box>
                              <Typography variant="caption" sx={{ display: 'block', fontStyle: 'italic', color: 'text.secondary' }}>
                                100% = Maximum expected muscle displacement for AU{au.au_number}
                              </Typography>
                              <Divider sx={{ my: 1 }} />
                              <Typography variant="caption" sx={{ display: 'block' }}>
                                Click card to see all {au.activation_count} frames with timestamps
                              </Typography>
                            </Box>
                          }
                          arrow
                          placement="top"
                        >
                          <Chip 
                            label={`Max: ${(au.max_intensity * 100).toFixed(1)}%`}
                            size="small"
                            sx={{ bgcolor: getIntensityColor(au.max_intensity), color: 'white', cursor: 'help' }}
                          />
                        </Tooltip>
                        <Tooltip 
                          title={
                            <Box sx={{ maxWidth: 400 }}>
                              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                Avg Intensity: {(au.avg_intensity * 100).toFixed(1)}%
                              </Typography>
                              <Typography variant="caption" sx={{ display: 'block', mb: 1, fontFamily: 'monospace', bgcolor: 'rgba(0,0,0,0.1)', p: 0.5, borderRadius: 0.5 }}>
                                Calculation:<br/>
                                Sum of all intensities / {au.activation_count} frames<br/>
                                = {(au.avg_intensity * 100).toFixed(1)}%
                              </Typography>
                              <Typography variant="caption" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                                Average across all frames where AU{au.au_number} was active
                              </Typography>
                            </Box>
                          }
                          arrow
                          placement="top"
                        >
                          <Chip 
                            label={`Avg: ${(au.avg_intensity * 100).toFixed(1)}%`}
                            size="small"
                            variant="outlined"
                            sx={{ cursor: 'help' }}
                          />
                        </Tooltip>
                        <Chip 
                          label={`${au.activation_count} activations`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    </Box>
                    <Box textAlign="center" sx={{ ml: 2 }}>
                      <Typography variant="caption" color="textSecondary">
                        Click for details
                      </Typography>
                      <br />
                      <AccessTimeIcon color="action" />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Clinical Notes */}
      {auData.clinical_notes && (
        <Paper sx={{ p: 3, mt: 3, bgcolor: 'info.light' }}>
          <Typography variant="h6" gutterBottom>
            Clinical Notes
          </Typography>
          <Typography variant="body2">
            Action Units represent <strong>objective muscle movements</strong> not emotions. 
            For example, AU12 (Mouth Corners Pulled Up) indicates the physical action of smiling, 
            but happens in both genuine smiling and social masking. 
            Interpret these patterns within the full clinical context.
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            {auData.clinical_notes}
          </Typography>
        </Paper>
      )}

      {/* AU Details Dialog */}
      <Dialog 
        open={auDetailsDialog} 
        onClose={() => setAuDetailsDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {selectedAU && (
            <Box>
              <Typography variant="h5">
                AU{selectedAU}: {AU_DESCRIPTIONS[selectedAU]?.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {AU_DESCRIPTIONS[selectedAU]?.description}
              </Typography>
            </Box>
          )}
        </DialogTitle>
        
        <DialogContent>
          {loadingEvidence ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
              <CircularProgress />
              <Typography variant="body1" sx={{ ml: 2 }}>
                Loading evidence frames...
              </Typography>
            </Box>
          ) : (
            <Box>
              {/* Video Player */}
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.900' }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                  📹 Video Player
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mb: 2, color: 'grey.400' }}>
                  Click any timestamp below to jump to that moment in the video
                </Typography>
                <video
                  ref={(ref) => setVideoPlayerRef(ref)}
                  controls
                  style={{ width: '100%', maxHeight: '400px', borderRadius: '8px' }}
                  src={`http://localhost:8000/sessions/${sessionId}/video`}
                >
                  Your browser does not support the video tag.
                </video>
              </Paper>

              {/* AU Information */}
              {selectedAU && AU_DESCRIPTIONS[selectedAU] && (
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">Action Unit Information</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2">
                          <strong>Muscle:</strong> {AU_DESCRIPTIONS[selectedAU].muscle}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          <strong>Clinical Significance:</strong> {AU_DESCRIPTIONS[selectedAU].clinical}
                        </Typography>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Time-stamped Evidence */}
              <Accordion defaultExpanded sx={{ mt: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6">
                    Time-stamped Evidence ({frameEvidence.length} instances)
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {frameEvidence.length > 0 ? (
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell><strong>Time</strong></TableCell>
                            <TableCell><strong>Frame</strong></TableCell>
                            <TableCell>
                              <Tooltip
                                title={
                                  <Box sx={{ p: 1, maxWidth: 400 }}>
                                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                      Intensity Calculation
                                    </Typography>
                                    <Box sx={{ fontFamily: 'monospace', fontSize: '0.75rem', bgcolor: 'rgba(0,0,0,0.1)', p: 1, borderRadius: 1, mb: 1 }}>
                                      1. Measure landmark displacement<br/>
                                      2. Normalize by face size<br/>
                                      3. intensity = (measured - baseline) / range<br/>
                                      4. Clip to 0-100%
                                    </Box>
                                    <Typography variant="caption" sx={{ fontStyle: 'italic' }}>
                                      100% = Maximum expected muscle displacement
                                    </Typography>
                                  </Box>
                                }
                                arrow
                              >
                                <strong style={{ cursor: 'help' }}>Intensity</strong>
                              </Tooltip>
                            </TableCell>
                            <TableCell>
                              <Tooltip
                                title={
                                  <Box sx={{ p: 1, maxWidth: 450 }}>
                                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                      Confidence Calculation
                                    </Typography>
                                    <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>
                                      <strong>What it measures:</strong> Reliability of the AU detection
                                    </Typography>
                                    <Box sx={{ fontFamily: 'monospace', fontSize: '0.75rem', bgcolor: 'rgba(0,0,0,0.1)', p: 1, borderRadius: 1, mb: 1 }}>
                                      <strong>Formula:</strong><br/>
                                      confidence = base_confidence × landmark_visibility<br/>
                                      <br/>
                                      <strong>Components:</strong><br/>
                                      • base_confidence: AU-specific reliability (0.70-0.90)<br/>
                                      &nbsp;&nbsp;- Subtle AUs (AU4): 0.70<br/>
                                      &nbsp;&nbsp;- Standard AUs: 0.75-0.85<br/>
                                      &nbsp;&nbsp;- Reliable AUs (AU26): 0.90<br/>
                                      <br/>
                                      • landmark_visibility: MediaPipe visibility score<br/>
                                      &nbsp;&nbsp;- Average of all facial landmark visibility<br/>
                                      &nbsp;&nbsp;- Affected by occlusion, lighting, head pose<br/>
                                      &nbsp;&nbsp;- Range: 0.0 (occluded) to 1.0 (fully visible)
                                    </Box>
                                    <Typography variant="caption" sx={{ fontStyle: 'italic' }}>
                                      Example: AU1 (base=0.85) × visibility (0.88) = 75% confidence
                                    </Typography>
                                  </Box>
                                }
                                arrow
                              >
                                <strong style={{ cursor: 'help' }}>Confidence</strong>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {frameEvidence.slice(0, 50).map((evidence, idx) => (
                            <TableRow 
                              key={idx} 
                              hover
                              onClick={() => handleTimestampClick(evidence.timestamp)}
                              sx={{ 
                                cursor: 'pointer',
                                '&:hover': { bgcolor: 'action.selected' }
                              }}
                            >
                              <TableCell>
                                <Tooltip title="Click to jump to this moment in the video" arrow>
                                  <Chip 
                                    label={formatTime(evidence.timestamp)}
                                    size="small"
                                    icon={<AccessTimeIcon />}
                                    sx={{ cursor: 'pointer' }}
                                  />
                                </Tooltip>
                              </TableCell>
                              <TableCell>{evidence.frame_number}</TableCell>
                              <TableCell>
                                <Chip 
                                  label={`${(evidence.intensity * 100).toFixed(1)}%`}
                                  size="small"
                                  sx={{ 
                                    bgcolor: getIntensityColor(evidence.intensity), 
                                    color: 'white' 
                                  }}
                                />
                              </TableCell>
                              <TableCell>
                                <Chip 
                                  label={`${(evidence.confidence * 100).toFixed(0)}%`}
                                  size="small"
                                  variant="outlined"
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  ) : (
                    <Alert severity="info">
                      No evidence frames found for this Action Unit.
                    </Alert>
                  )}
                  
                  {frameEvidence.length > 50 && (
                    <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
                      Showing first 50 instances. Total: {frameEvidence.length}
                    </Typography>
                  )}
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setAuDetailsDialog(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Formula Explanation Dialog */}
      <Dialog 
        open={formulaDialog} 
        onClose={() => setFormulaDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {selectedMetric && getMetricFormula(selectedMetric) && (
            <Box>
              <Typography variant="h5">
                {getMetricFormula(selectedMetric).name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                How this score is calculated
              </Typography>
            </Box>
          )}
        </DialogTitle>
        
        <DialogContent>
          {selectedMetric && getMetricFormula(selectedMetric) && (
            <Box>
              {/* Formula */}
              <Paper sx={{ p: 2, mb: 3, bgcolor: 'grey.100' }}>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  Formula:
                </Typography>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    fontFamily: 'monospace', 
                    fontSize: '0.95rem',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                  }}
                >
                  {getMetricFormula(selectedMetric).formula}
                </Typography>
              </Paper>

              {/* Components Breakdown */}
              <Typography variant="h6" gutterBottom>
                Components:
              </Typography>
              <List>
                {getMetricFormula(selectedMetric).components.map((component, idx) => (
                  <React.Fragment key={idx}>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Box display="flex" justifyContent="space-between" alignItems="center">
                            <Typography variant="subtitle1" fontWeight="bold">
                              {component.name}
                            </Typography>
                            {component.name === 'Total Possible AUs' ? (
                              <Tooltip 
                                title={<All15AUsTooltip />}
                                arrow
                                placement="left"
                              >
                                <Chip 
                                  label={component.value} 
                                  color="primary" 
                                  size="small"
                                  sx={{ cursor: 'help' }}
                                />
                              </Tooltip>
                            ) : component.name === 'Unique AUs Detected' ? (
                              <Tooltip 
                                title={<DetectedAUsTooltip detectedAUs={auData?.most_activated_units?.map(au => au.au_number) || []} />}
                                arrow
                                placement="left"
                              >
                                <Chip 
                                  label={component.value} 
                                  color="primary" 
                                  size="small"
                                  sx={{ cursor: 'help' }}
                                />
                              </Tooltip>
                            ) : component.name === 'Left-Right Balance' ? (
                              <Tooltip 
                                title={<SymmetryExplanationTooltip symmetryData={auData?.calculation_details?.symmetry} />}
                                arrow
                                placement="left"
                              >
                                <Chip 
                                  label={component.value} 
                                  color="primary" 
                                  size="small"
                                  sx={{ cursor: 'help' }}
                                />
                              </Tooltip>
                            ) : (
                              <Chip 
                                label={component.value} 
                                color="primary" 
                                size="small"
                              />
                            )}
                          </Box>
                        }
                        secondary={
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2" color="textSecondary">
                              {component.description}
                            </Typography>
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                mt: 1, 
                                fontFamily: 'monospace',
                                bgcolor: 'grey.50',
                                p: 1,
                                borderRadius: 1
                              }}
                            >
                              {component.calculation}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                    {idx < getMetricFormula(selectedMetric).components.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>

              {/* RAW DATA SECTION - NEW */}
              {auData?.calculation_details && (
                <Accordion sx={{ mt: 3, bgcolor: 'warning.light' }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6" color="warning.dark">
                      🔬 Raw Data & Frame-by-Frame Analysis
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Alert severity="info" sx={{ mb: 2 }}>
                        This section shows the actual raw data used in calculations. 
                        Every number is traceable to specific video frames.
                      </Alert>

                      {/* Data Quality Metrics */}
                      {auData.calculation_details.data_quality && (
                        <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                            📊 Data Quality Metrics
                          </Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>Total Frames Analyzed:</strong> {auData.calculation_details.data_quality.total_frames_analyzed}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>Frames with Face:</strong> {auData.calculation_details.data_quality.frames_with_face_detected}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography variant="body2">
                                <strong>Inactive Frames:</strong> {auData.calculation_details.data_quality.inactive_frames}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Tooltip 
                                title={
                                  <Box sx={{ p: 1 }}>
                                    <Typography variant="body2" fontWeight="bold" gutterBottom>
                                      What are "Total AU Activations"?
                                    </Typography>
                                    <Typography variant="body2" sx={{ mb: 1 }}>
                                      This is the SUM of all AU activations across ALL frames.
                                    </Typography>
                                    <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem', bgcolor: 'rgba(255,255,255,0.1)', p: 1, borderRadius: 1 }}>
                                      Example:<br/>
                                      Frame 1: 3 AUs active<br/>
                                      Frame 2: 4 AUs active<br/>
                                      Frame 3: 2 AUs active<br/>
                                      ...<br/>
                                      Total = 3+4+2+... = {auData.calculation_details.data_quality.total_au_activations}
                                    </Typography>
                                    <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                                      Average per frame: {(auData.calculation_details.data_quality.total_au_activations / auData.calculation_details.data_quality.total_frames_analyzed).toFixed(1)} AUs/frame
                                    </Typography>
                                  </Box>
                                }
                                arrow
                                placement="top"
                              >
                                <Typography variant="body2" sx={{ cursor: 'help', display: 'inline-block' }}>
                                  <strong>Total AU Activations:</strong> {auData.calculation_details.data_quality.total_au_activations}
                                </Typography>
                              </Tooltip>
                            </Grid>
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Unique AUs Detected:</strong> {auData.calculation_details.data_quality.unique_aus_detected}
                              </Typography>
                            </Grid>
                          </Grid>
                        </Paper>
                      )}

                      {/* AU Activation Details */}
                      {auData.most_activated_units && auData.most_activated_units.length > 0 && (
                        <Paper sx={{ p: 2, mb: 2, bgcolor: 'info.light' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                            🎯 Action Unit Activation Details
                          </Typography>
                          <Typography variant="caption" color="textSecondary" display="block" sx={{ mb: 1 }}>
                            Each AU represents a specific facial muscle movement. These are the raw measurements:
                          </Typography>
                          <TableContainer>
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell><strong>AU</strong></TableCell>
                                  <TableCell><strong>Name</strong></TableCell>
                                  <TableCell><strong>Activations</strong></TableCell>
                                  <TableCell>
                                    <Tooltip 
                                      title={
                                        <Box sx={{ p: 1 }}>
                                          <Typography variant="body2" fontWeight="bold">Max Intensity</Typography>
                                          <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
                                            The HIGHEST intensity value recorded for this AU across all frames.
                                          </Typography>
                                          <Typography variant="caption" sx={{ display: 'block', mt: 1, fontFamily: 'monospace', bgcolor: 'rgba(255,255,255,0.1)', p: 0.5 }}>
                                            Example: AU1 in 29 frames<br/>
                                            Frame 1: 95%<br/>
                                            Frame 2: 100%<br/>
                                            Frame 3: 98%<br/>
                                            ...<br/>
                                            Max = 100% (highest)
                                          </Typography>
                                        </Box>
                                      }
                                      arrow
                                    >
                                      <strong style={{ cursor: 'help' }}>Max Intensity</strong>
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell>
                                    <Tooltip 
                                      title={
                                        <Box sx={{ p: 1 }}>
                                          <Typography variant="body2" fontWeight="bold">Avg Intensity</Typography>
                                          <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
                                            The AVERAGE intensity across all frames where this AU was active.
                                          </Typography>
                                          <Typography variant="caption" sx={{ display: 'block', mt: 1, fontFamily: 'monospace', bgcolor: 'rgba(255,255,255,0.1)', p: 0.5 }}>
                                            Example: AU1 in 29 frames<br/>
                                            Frame 1: 95%<br/>
                                            Frame 2: 100%<br/>
                                            Frame 3: 98%<br/>
                                            ...<br/>
                                            Avg = (95+100+98+...)/29
                                          </Typography>
                                        </Box>
                                      }
                                      arrow
                                    >
                                      <strong style={{ cursor: 'help' }}>Avg Intensity</strong>
                                    </Tooltip>
                                  </TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {auData.most_activated_units.map((au) => (
                                  <TableRow 
                                    key={au.au_number} 
                                    hover
                                    onClick={() => handleAUClick(au.au_number)}
                                    sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                                  >
                                    <TableCell>AU{au.au_number}</TableCell>
                                    <TableCell>
                                      <Typography variant="caption">
                                        {AU_DESCRIPTIONS[au.au_number]?.name || 'Unknown'}
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Tooltip 
                                        title={
                                          <Box>
                                            <Typography variant="caption">
                                              AU{au.au_number} was detected in {au.activation_count} out of {auData.total_frames_analyzed} frames
                                            </Typography>
                                            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontStyle: 'italic' }}>
                                              Click row to see timestamps for all frames
                                            </Typography>
                                          </Box>
                                        }
                                        arrow
                                      >
                                        <Chip 
                                          label={`${au.activation_count} frames`}
                                          size="small"
                                          variant="outlined"
                                          sx={{ cursor: 'pointer' }}
                                        />
                                      </Tooltip>
                                    </TableCell>
                                    <TableCell>
                                      <Tooltip 
                                        title={
                                          <Box sx={{ maxWidth: 450 }}>
                                            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                              Max Intensity: {(au.max_intensity * 100).toFixed(1)}%
                                            </Typography>
                                            <Divider sx={{ my: 1 }} />
                                            <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>
                                              <strong>How intensity is calculated:</strong>
                                            </Typography>
                                            <Box sx={{ fontFamily: 'monospace', fontSize: '0.75rem', bgcolor: 'rgba(0,0,0,0.1)', p: 1, borderRadius: 1, mb: 1 }}>
                                              1. Measure landmark displacement (pixels)<br/>
                                              2. Normalize by face size<br/>
                                              3. Compare to neutral baseline<br/>
                                              4. intensity = (measured - baseline) / range<br/>
                                              5. Clip to 0-100%
                                            </Box>
                                            <Typography variant="caption" sx={{ display: 'block', fontStyle: 'italic', color: 'text.secondary' }}>
                                              100% = Maximum expected muscle displacement for this AU
                                            </Typography>
                                            <Divider sx={{ my: 1 }} />
                                            <Typography variant="caption" sx={{ display: 'block' }}>
                                              Click row to see frame-by-frame measurements with timestamps
                                            </Typography>
                                          </Box>
                                        }
                                        arrow
                                        placement="top"
                                      >
                                        <Chip 
                                          label={`${(au.max_intensity * 100).toFixed(1)}%`}
                                          size="small"
                                          sx={{ bgcolor: getIntensityColor(au.max_intensity), color: 'white', cursor: 'help' }}
                                        />
                                      </Tooltip>
                                    </TableCell>
                                    <TableCell>
                                      <Tooltip 
                                        title={
                                          <Box sx={{ maxWidth: 400 }}>
                                            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                                              Avg Intensity: {(au.avg_intensity * 100).toFixed(1)}%
                                            </Typography>
                                            <Typography variant="caption" sx={{ display: 'block', mb: 1, fontFamily: 'monospace', bgcolor: 'rgba(0,0,0,0.1)', p: 0.5, borderRadius: 0.5 }}>
                                              Calculation:<br/>
                                              Sum of all intensities / {au.activation_count} frames<br/>
                                              = {(au.avg_intensity * 100).toFixed(1)}%
                                            </Typography>
                                            <Typography variant="caption" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                                              Click AU{au.au_number} row to see frame-by-frame breakdown with timestamps
                                            </Typography>
                                          </Box>
                                        }
                                        arrow
                                        placement="top"
                                      >
                                        <span style={{ cursor: 'help' }}>
                                          {(au.avg_intensity * 100).toFixed(1)}%
                                        </span>
                                      </Tooltip>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        </Paper>
                      )}

                      {/* Calculation Trace */}
                      {selectedMetric && auData.calculation_details[selectedMetric] && (
                        <Paper sx={{ p: 2, bgcolor: 'success.light' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom color="success.dark">
                            🧮 Detailed Calculation Trace
                          </Typography>
                          <Typography variant="caption" color="textSecondary" display="block" sx={{ mb: 2 }}>
                            This shows exactly how the raw data was transformed into the final score:
                          </Typography>
                          
                          {/* Show all inputs with explanations */}
                          <Box sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                            {Object.entries(auData.calculation_details[selectedMetric].inputs || {}).map(([key, value]) => (
                              <Box key={key} sx={{ mb: 1, p: 1, bgcolor: 'white', borderRadius: 1 }}>
                                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                                  <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : JSON.stringify(value)}
                                </Typography>
                                <Typography variant="caption" color="textSecondary">
                                  {getInputExplanation(key, value)}
                                </Typography>
                              </Box>
                            ))}
                          </Box>

                          {/* Show calculation steps */}
                          {auData.calculation_details[selectedMetric].step_by_step && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="body2" fontWeight="bold" gutterBottom>
                                Step-by-Step Calculation:
                              </Typography>
                              {auData.calculation_details[selectedMetric].step_by_step.map((step, idx) => (
                                <Typography 
                                  key={idx}
                                  variant="body2" 
                                  sx={{ 
                                    fontFamily: 'monospace',
                                    ml: 2,
                                    color: 'success.dark'
                                  }}
                                >
                                  {idx + 1}. {step}
                                </Typography>
                              ))}
                            </Box>
                          )}
                        </Paper>
                      )}

                      {/* Database Query */}
                      <Paper sx={{ p: 2, mt: 2, bgcolor: 'grey.100' }}>
                        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                          💾 Database Verification Query
                        </Typography>
                        <Typography variant="caption" color="textSecondary" display="block" sx={{ mb: 1 }}>
                          Run this SQL query to verify the raw data:
                        </Typography>
                        <Box sx={{ 
                          fontFamily: 'monospace', 
                          fontSize: '0.75rem',
                          bgcolor: 'black',
                          color: 'lime',
                          p: 2,
                          borderRadius: 1,
                          overflow: 'auto'
                        }}>
                          <pre style={{ margin: 0 }}>
{`-- Verify AU activations
SELECT 
    au_number,
    COUNT(*) as activation_count,
    AVG(au_intensity) as avg_intensity,
    MAX(au_intensity) as max_intensity
FROM frame_analysis
WHERE session_id = '${sessionId}'
  AND au_present = 1
GROUP BY au_number
ORDER BY activation_count DESC;

-- Verify frame count
SELECT COUNT(DISTINCT frame_number) as total_frames
FROM frame_analysis
WHERE session_id = '${sessionId}';

-- Verify unique AUs
SELECT COUNT(DISTINCT au_number) as unique_aus
FROM frame_analysis
WHERE session_id = '${sessionId}'
  AND au_present = 1;`}
                          </pre>
                        </Box>
                      </Paper>

                      {/* Audit Trail */}
                      {auData.calculation_details.audit_trail && (
                        <Paper sx={{ p: 2, mt: 2, bgcolor: 'info.light' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                            📋 Audit Trail
                          </Typography>
                          <Typography variant="body2">
                            <strong>Source Code:</strong> {auData.calculation_details.audit_trail.source_code}
                          </Typography>
                          <Typography variant="body2">
                            <strong>Database Table:</strong> {auData.calculation_details.audit_trail.database_table}
                          </Typography>
                          <Typography variant="body2">
                            <strong>Session ID:</strong> {auData.calculation_details.audit_trail.session_id}
                          </Typography>
                          <Typography variant="body2">
                            <strong>Documentation:</strong> {auData.calculation_details.audit_trail.documentation}
                          </Typography>
                        </Paper>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Final Calculation */}
              <Paper sx={{ p: 2, mt: 3, bgcolor: 'success.light' }}>
                <Typography variant="subtitle2" color="success.dark" gutterBottom>
                  Final Calculation:
                </Typography>
                <Typography 
                  variant="body1" 
                  sx={{ 
                    fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    color: 'success.dark'
                  }}
                >
                  {getMetricFormula(selectedMetric).finalCalculation}
                </Typography>
              </Paper>

              {/* Interpretation */}
              <Paper sx={{ p: 2, mt: 3, bgcolor: 'info.light' }}>
                <Typography variant="subtitle2" color="info.dark" gutterBottom>
                  Interpretation:
                </Typography>
                <Typography variant="body2" color="info.dark">
                  {getMetricFormula(selectedMetric).interpretation}
                </Typography>
              </Paper>
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setFormulaDialog(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default FacialActionUnitsViewer;