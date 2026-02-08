import React from 'react';
import Brain19Visualizer from './Brain19Visualizer';

// Sample snapshot data for demonstration
const sampleSnapshot = {
  stm: {
    context_id: 1,
    active_concepts: [
      { concept_id: 1, activation: 0.95 },
      { concept_id: 2, activation: 0.85 },
      { concept_id: 3, activation: 0.72 },
      { concept_id: 4, activation: 0.58 },
      { concept_id: 5, activation: 0.45 },
      { concept_id: 6, activation: 0.33 }
    ],
    active_relations: [
      { source: 1, target: 2, type: 'IS_A', activation: 0.88 },
      { source: 2, target: 3, type: 'HAS_PROPERTY', activation: 0.75 },
      { source: 3, target: 4, type: 'CAUSES', activation: 0.62 },
      { source: 1, target: 5, type: 'SIMILAR_TO', activation: 0.55 },
      { source: 4, target: 6, type: 'ENABLES', activation: 0.40 }
    ]
  },
  concepts: [
    { id: 1, label: 'Cat', epistemic_type: 'FACT', trust: 0.98 },
    { id: 2, label: 'Animal', epistemic_type: 'FACT', trust: 0.99 },
    { id: 3, label: 'Furry', epistemic_type: 'FACT', trust: 0.92 },
    { id: 4, label: 'Meow', epistemic_type: 'THEORY', trust: 0.78 },
    { id: 5, label: 'Dog', epistemic_type: 'FACT', trust: 0.97 },
    { id: 6, label: 'Sound', epistemic_type: 'HYPOTHESIS', trust: 0.65 }
  ],
  curiosity_triggers: [
    { 
      type: 'SHALLOW_RELATIONS', 
      description: 'Many concepts activated but few relations' 
    },
    { 
      type: 'LOW_EXPLORATION', 
      description: 'Stable context with minimal variation' 
    }
  ]
};

const App = () => {
  return <Brain19Visualizer snapshot={sampleSnapshot} />;
};

export default App;
