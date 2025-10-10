import React from 'react';
import { Row, Col, Card, Statistic, Table, Alert, Spin } from 'antd';
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined, 
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Dashboard = () => {
  // Mock data - in real app, this would come from API
  const stats = [
    {
      title: 'Total Transactions',
      value: 12543,
      precision: 0,
      valueStyle: { color: '#1890ff' },
      prefix: <ArrowUpOutlined />,
    },
    {
      title: 'Fraud Detected',
      value: 89,
      precision: 0,
      valueStyle: { color: '#cf1322' },
      prefix: <ExclamationCircleOutlined />,
    },
    {
      title: 'Blocked Transactions',
      value: 67,
      precision: 0,
      valueStyle: { color: '#cf1322' },
      prefix: <ArrowDownOutlined />,
    },
    {
      title: 'Challenged Transactions',
      value: 156,
      precision: 0,
      valueStyle: { color: '#faad14' },
      prefix: <ClockCircleOutlined />,
    },
    {
      title: 'False Positives',
      value: 12,
      precision: 0,
      valueStyle: { color: '#52c41a' },
      prefix: <CheckCircleOutlined />,
    },
    {
      title: 'Accuracy Rate',
      value: 98.5,
      precision: 1,
      suffix: '%',
      valueStyle: { color: '#52c41a' },
      prefix: <CheckCircleOutlined />,
    },
  ];

  const fraudTrendData = [
    { time: '00:00', fraud: 2, normal: 45 },
    { time: '04:00', fraud: 1, normal: 23 },
    { time: '08:00', fraud: 3, normal: 67 },
    { time: '12:00', fraud: 5, normal: 89 },
    { time: '16:00', fraud: 4, normal: 78 },
    { time: '20:00', fraud: 6, normal: 92 },
  ];

  const decisionData = [
    { name: 'Allowed', value: 12300, color: '#52c41a' },
    { name: 'Challenged', value: 156, color: '#faad14' },
    { name: 'Blocked', value: 67, color: '#cf1322' },
  ];

  const recentTransactions = [
    {
      key: '1',
      transactionId: 'TXN123456789',
      amount: '₹15,000',
      merchant: 'Amazon India',
      riskScore: 0.85,
      decision: 'BLOCKED',
      time: '2 min ago',
    },
    {
      key: '2',
      transactionId: 'TXN123456790',
      amount: '₹2,500',
      merchant: 'Swiggy',
      riskScore: 0.25,
      decision: 'ALLOWED',
      time: '5 min ago',
    },
    {
      key: '3',
      transactionId: 'TXN123456791',
      amount: '₹50,000',
      merchant: 'Crypto Exchange',
      riskScore: 0.92,
      decision: 'CHALLENGED',
      time: '8 min ago',
    },
  ];

  const transactionColumns = [
    {
      title: 'Transaction ID',
      dataIndex: 'transactionId',
      key: 'transactionId',
    },
    {
      title: 'Amount',
      dataIndex: 'amount',
      key: 'amount',
    },
    {
      title: 'Merchant',
      dataIndex: 'merchant',
      key: 'merchant',
    },
    {
      title: 'Risk Score',
      dataIndex: 'riskScore',
      key: 'riskScore',
      render: (score) => (
        <span className={score > 0.7 ? 'risk-score-high' : score > 0.4 ? 'risk-score-medium' : 'risk-score-low'}>
          {(score * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      title: 'Decision',
      dataIndex: 'decision',
      key: 'decision',
      render: (decision) => (
        <span className={`decision-${decision.toLowerCase()}`}>
          {decision}
        </span>
      ),
    },
    {
      title: 'Time',
      dataIndex: 'time',
      key: 'time',
    },
  ];

  return (
    <div>
      <Row gutter={[16, 16]}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} lg={8} xl={4} key={index}>
            <Card>
              <Statistic
                title={stat.title}
                value={stat.value}
                precision={stat.precision}
                valueStyle={stat.valueStyle}
                prefix={stat.prefix}
                suffix={stat.suffix}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={16}>
          <Card title="Fraud Trend (Last 24 Hours)" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={fraudTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="fraud" stroke="#cf1322" strokeWidth={2} name="Fraud" />
                <Line type="monotone" dataKey="normal" stroke="#52c41a" strokeWidth={2} name="Normal" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          <Card title="Decision Distribution" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={decisionData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {decisionData.map((entry, index) => (
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
        <Col xs={24}>
          <Card title="Recent Transactions" size="small">
            <Table
              columns={transactionColumns}
              dataSource={recentTransactions}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <Alert
            message="System Status"
            description="All systems operational. Last model update: 2 hours ago. Next scheduled retraining: 6 hours."
            type="success"
            showIcon
          />
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
