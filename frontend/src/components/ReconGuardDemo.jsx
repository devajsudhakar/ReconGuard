import React, { useState } from 'react';

const ReconGuardDemo = () => {
  const [formData, setFormData] = useState({
    user_id: 'user_123',
    merchant_id: 'merch_456',
    amount: 100.0,
    timestamp: new Date().toISOString()
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h1 className="text-2xl font-bold text-center text-gray-800">ReconGuard Demo</h1>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">User ID</label>
          <input
            type="text"
            name="user_id"
            value={formData.user_id}
            onChange={handleChange}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">Merchant ID</label>
          <input
            type="text"
            name="merchant_id"
            value={formData.merchant_id}
            onChange={handleChange}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">Amount</label>
          <input
            type="number"
            name="amount"
            value={formData.amount}
            onChange={handleChange}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">Timestamp</label>
          <input
            type="text"
            name="timestamp"
            value={formData.timestamp}
            onChange={handleChange}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {loading ? 'Analyzing...' : 'Check for Anomaly'}
        </button>
      </form>

      {error && (
        <div className="p-4 bg-red-50 text-red-700 rounded-md">
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className={`p-4 rounded-md ${result.is_anomaly ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
          <h2 className={`text-lg font-semibold ${result.is_anomaly ? 'text-red-800' : 'text-green-800'}`}>
            {result.is_anomaly ? 'Anomaly Detected!' : 'Transaction Normal'}
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            <strong>Anomaly Score:</strong> {result.anomaly_score.toFixed(4)}
          </p>
          {result.reasons && result.reasons.length > 0 && (
            <div className="mt-2">
              <p className="text-sm font-medium text-gray-700">Reasons:</p>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {result.reasons.map((reason, index) => (
                  <li key={index}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ReconGuardDemo;
