import React from 'react';

export const Card = ({ className = '', children, ...props }) => (
  <div className={`rounded-lg border border-gray-200 bg-white shadow-sm ${className}`} {...props}>
    {children}
  </div>
);

export const CardHeader = ({ className = '', children, ...props }) => (
  <div className={`border-b border-gray-200 px-6 py-4 ${className}`} {...props}>
    {children}
  </div>
);

export const CardTitle = ({ className = '', children, ...props }) => (
  <h2 className={`text-lg font-semibold text-gray-900 ${className}`} {...props}>
    {children}
  </h2>
);

export const CardContent = ({ className = '', children, ...props }) => (
  <div className={`px-6 py-4 ${className}`} {...props}>
    {children}
  </div>
);

export const Tabs = ({ defaultValue, children, className = '', ...props }) => {
  const [activeTab, setActiveTab] = React.useState(defaultValue);
  
  return (
    <div className={`w-full ${className}`} {...props}>
      {React.Children.map(children, child => 
        React.cloneElement(child, { activeTab, setActiveTab })
      )}
    </div>
  );
};

export const TabsList = ({ children, className = '', activeTab, setActiveTab, ...props }) => (
  <div className={`flex gap-2 border-b border-gray-200 ${className}`} {...props}>
    {children}
  </div>
);

export const TabsTrigger = ({ value, children, className = '', activeTab, setActiveTab, ...props }) => (
  <button
    onClick={() => setActiveTab?.(value)}
    className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
      activeTab === value
        ? 'border-blue-500 text-blue-600'
        : 'border-transparent text-gray-600 hover:text-gray-900'
    } ${className}`}
    {...props}
  >
    {children}
  </button>
);

export const TabsContent = ({ value, children, className = '', activeTab, ...props }) => (
  activeTab === value ? (
    <div className={`mt-4 ${className}`} {...props}>
      {children}
    </div>
  ) : null
);

export const Alert = ({ className = '', children, variant = 'default', ...props }) => {
  const variantClasses = {
    default: 'bg-blue-50 border-blue-200 text-blue-900',
    destructive: 'bg-red-50 border-red-200 text-red-900',
    success: 'bg-green-50 border-green-200 text-green-900',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-900'
  };
  
  return (
    <div 
      className={`rounded-lg border px-4 py-3 ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export const AlertDescription = ({ className = '', children, ...props }) => (
  <p className={`text-sm ${className}`} {...props}>
    {children}
  </p>
);

export const Badge = ({ className = '', variant = 'default', children, ...props }) => {
  const variantClasses = {
    default: 'bg-gray-100 text-gray-800',
    destructive: 'bg-red-100 text-red-800',
    success: 'bg-green-100 text-green-800',
    secondary: 'bg-blue-100 text-blue-800'
  };
  
  return (
    <span 
      className={`inline-block rounded-full px-3 py-1 text-xs font-semibold ${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </span>
  );
};

export const Progress = ({ value = 0, className = '' }) => (
  <div className={`h-2 w-full bg-gray-200 rounded-full overflow-hidden ${className}`}>
    <div
      className="h-full bg-blue-500 transition-all duration-300"
      style={{ width: `${Math.min(value, 100)}%` }}
    />
  </div>
);
