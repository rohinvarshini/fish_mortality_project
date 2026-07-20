import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { ModelDashboard } from './pages/ModelDashboard';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<ModelDashboard />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
