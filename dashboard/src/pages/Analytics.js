import React from 'react';
import { Card, Row, Col, Statistic, Select, DatePicker } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const { RangePicker } = DatePicker;
const { Option } = Select;

const Analytics = () => {
  // Mock data for analytics
  const fraudTrendData = [
    { date: '2024-01-01', fraud: 12, normal: 1200, blocked: 8, challenged: 15 },
    { date: '2024-01-02', fraud: 15, normal: 1350, blocked: 10, challenged: 18 },
    { date: '2024-01-03', fraud: 8, normal: 1100, blocked: 5, challenged: 12 },
    { date: '2024-01-04', fraud: 20, normal: 1400, blocked: 15, challenged: 25 },
    { date: '2024-01-05', fraud: 18, normal: 1300, blocked: 12, challenged: 20 },
    { date: '2024-01-06', fraud: 14, normal: 1250, blocked: 9, challenged: 16 },
    { date: '2024-01-07', fraud: 16, normal: 1380, blocked: 11, challenged: 19 },
  ];

  const merchantFraudData = [
    { merchant: 'Amazon', transactions: 500, fraud: 5, fraudRate: 1.0 },
    { merchant: 'Swiggy', transactions: 800, fraud: 2, fraudRate: 0.25 },
    { merchant: 'Crypto Exchange', transactions: 100, fraud: 15, fraudRate: 15.0 },
    { merchant: 'Uber', transactions: 600, fraud: 3, fraudRate: 0.5 },
    { merchant: 'Zomato', transactions: 700, fraud: 4, fraudRate: 0.57 },
    { merchant: 'Gambling Site', transactions: 50, fraud: 20, fraudRate: 40.0 },
  ];

  const hourlyFraudData = [
    { hour: '00:00', fraud: 2, normal: 45 },
    { hour: '02:00', fraud: 1, normal: 23 },
    { hour: '04:00', fraud: 0, normal: 12 },
    { hour: '06:00', fraud: 1, normal: 34 },
    { hour: '08:00', fraud: 3, normal: 67 },
    { hour: '10:00', fraud: 2, normal: 89 },
    { hour: '12:00', fraud: 5, normal: 120 },
    { hour: '14:00', fraud: 4, normal: 98 },
    { hour: '16:00', fraud: 3, normal: 78 },
    { hour: '18:00', fraud: 6, normal: 92 },
    { hour: '20:00', fraud: 8, normal: 110 },
    { hour: '22:00', fraud: 4, normal: 67 },
  ];

  const modelPerformanceData = [
    { model: 'XGBoost', accuracy: 94.5, precision: 92.1, recall: 89.3, f1: 90.7 },
    { model: 'LSTM', accuracy: 91.2, precision: 88.5, recall: 92.1, f1: 90.3 },
    { model: 'Autoencoder', accuracy: 89.8, precision: 85.2, recall: 94.5, f1: 89.6 },
    { model: 'GNN', accuracy: 92.1, precision: 90.3, recall: 87.8, f1: 89.0 },
    { model: 'Ensemble', accuracy: 96.8, precision: 95.2, recall: 93.7, f1: 94.4 },
  ];

  const riskDistributionData = [
    { name: 'Low Risk (0-0.3)', value: 8500, color: '#52c41a' },
    { name: 'Medium Risk (0.3-0.7)', value: 3200, color: '#faad14' },
    { name: 'High Risk (0.7-1.0)', value: 843, color: '#cf1322' },
  ];

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Transactions"
              value={12543}
              precision={0}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Fraud Rate"
              value={6.7}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Block Rate"
              value={4.2}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Accuracy"
              value={96.8}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="Fraud Trend Over Time" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={fraudTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="fraud" stroke="#cf1322" strokeWidth={2} name="Fraud" />
                <Line type="monotone" dataKey="normal" stroke="#52c41a" strokeWidth={2} name="Normal" />
                <Line type="monotone" dataKey="blocked" stroke="#faad14" strokeWidth={2} name="Blocked" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          <Card title="Risk Distribution" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={riskDistributionData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {riskDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="Fraud by Hour of Day" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={hourlyFraudData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="fraud" fill="#cf1322" name="Fraud" />
                <Bar dataKey="normal" fill="#52c41a" name="Normal" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card title="Merchant Fraud Rates" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={merchantFraudData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="merchant" type="category" width={100} />
                <Tooltip />
                <Bar dataKey="fraudRate" fill="#cf1322" name="Fraud Rate %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <Card title="Model Performance Comparison" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelPerformanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#1890ff" name="Accuracy %" />
                <Bar dataKey="precision" fill="#52c41a" name="Precision %" />
                <Bar dataKey="recall" fill="#faad14" name="Recall %" />
                <Bar dataKey="f1" fill="#cf1322" name="F1 Score %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Analytics;
