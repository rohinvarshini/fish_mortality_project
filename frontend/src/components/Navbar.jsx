import React from 'react';
import { Droplet, Activity } from 'lucide-react';

export function Navbar() {
  return (
    <header className="bg-black text-white sticky top-0 z-30 shadow-md">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-2 text-xl font-bold tracking-tight">
            <Droplet className="text-white" size={24} />
            <span>AquaGuard Model API</span>
          </div>
          
          <nav className="flex items-center space-x-2 border-l border-gray-700 pl-6">
            <div className="flex items-center px-4 py-2 bg-white text-black rounded-md text-sm font-medium">
              <Activity size={18} className="mr-2" />
              Live Inference Server
            </div>
          </nav>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm font-semibold">Local Environment</p>
            <p className="text-xs text-gray-400">bilstm_production.pt</p>
          </div>
          <div className="w-9 h-9 border-2 border-white rounded-full flex items-center justify-center font-bold text-sm bg-blue-600">
            ML
          </div>
        </div>
      </div>
    </header>
  );
}
