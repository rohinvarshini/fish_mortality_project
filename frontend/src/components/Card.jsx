import React from 'react';
import { cn } from '../lib/utils';
import { motion } from 'framer-motion';

export function Card({ className, children, ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "bg-white rounded-xl shadow-sm border border-slate-200 p-6",
        className
      )}
      {...props}
    >
      {children}
    </motion.div>
  );
}

export function CardHeader({ title, subtitle, className }) {
  return (
    <div className={cn("mb-5 pb-4 border-b border-slate-100", className)}>
      <h3 className="text-lg font-bold text-slate-800 tracking-tight">{title}</h3>
      {subtitle && <p className="text-sm font-medium text-slate-500 mt-1">{subtitle}</p>}
    </div>
  );
}
