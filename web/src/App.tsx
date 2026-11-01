import { Route, Routes } from 'react-router-dom'

import { Layout } from './components/Layout'
import { AgentWorkspace } from './pages/AgentWorkspace'
import { Benchmarks } from './pages/Benchmarks'
import { DataStudio } from './pages/DataStudio'
import { IdentityMaker } from './pages/IdentityMaker'
import { Hub } from './pages/Hub'
import { Machine } from './pages/Machine'
import { Models } from './pages/Models'
import { Overview } from './pages/Overview'
import { Playground } from './pages/Playground'
import { Runs } from './pages/Runs'
import { Settings } from './pages/Settings'
import { TrainingLab } from './pages/TrainingLab'
import { UpgradeLab } from './pages/UpgradeLab'

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="machine" element={<Machine />} />
        <Route path="models" element={<Models />} />
        <Route path="hub" element={<Hub />} />
        <Route path="playground" element={<Playground />} />
        <Route path="data" element={<DataStudio />} />
        <Route path="identity" element={<IdentityMaker />} />
        <Route path="training" element={<TrainingLab />} />
        <Route path="benchmarks" element={<Benchmarks />} />
        <Route path="upgrades" element={<UpgradeLab />} />
        <Route path="agent" element={<AgentWorkspace />} />
        <Route path="runs" element={<Runs />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
