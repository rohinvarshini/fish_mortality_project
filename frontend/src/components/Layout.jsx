import React from 'react';
import { Navbar } from './Navbar';
import { Outlet } from 'react-router-dom';

export function Layout() {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-black flex flex-col">
      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-6 md:p-8">
        <Outlet />
      </main>
    </div>
  );
}
