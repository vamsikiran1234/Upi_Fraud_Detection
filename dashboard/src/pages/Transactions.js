import React, { useState } from 'react';
import { Table, Card, Input, Select, DatePicker, Button, Space, Tag, Modal, Descriptions } from 'antd';
import { SearchOutlined, EyeOutlined, FilterOutlined } from '@ant-design/icons';
import ReactJson from 'react-json-view';

const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

const Transactions = () => {
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  // Mock data - in real app, this would come from API
  const transactions = [
    {
      key: '1',
      transactionId: 'TXN123456789',
      upiId: 'user@paytm',
      amount: 15000,
      merchant: 'Amazon India',
      merchantCategory: 'ecommerce',
      deviceId: 'device_123',
      ipAddress: '192.168.1.100',
      location: { lat: 28.6139, lon: 77.2090 },
      riskScore: 0.85,
      fraudProbability: 0.87,
      decision: 'BLOCKED',
      confidence: 0.92,
      processingTime: 45,
      timestamp: '2024-01-15 14:30:25',
      modelVersions: {
        xgboost: '1.0.0',
        lstm: '1.0.0',
        autoencoder: '1.0.0',
        gnn: '1.0.0'
      },
      explanation: {
        risk_factors: [
          { feature: 'amount', impact: 0.3, direction: 'increases', severity: 'high' },
          { feature: 'merchant_category', impact: 0.2, direction: 'increases', severity: 'medium' },
          { feature: 'device_risk_score', impact: 0.15, direction: 'increases', severity: 'medium' }
        ],
        human_readable: 'High transaction amount (₹15,000) increases fraud risk. E-commerce merchant category increases fraud risk. Suspicious device characteristics increase fraud risk.'
      },
      alerts: ['High amount transaction', 'Suspicious device detected']
    },
    {
      key: '2',
      transactionId: 'TXN123456790',
      upiId: 'user@paytm',
      amount: 2500,
      merchant: 'Swiggy',
      merchantCategory: 'food',
      deviceId: 'device_456',
      ipAddress: '192.168.1.101',
      location: { lat: 28.6139, lon: 77.2090 },
      riskScore: 0.25,
      fraudProbability: 0.23,
      decision: 'ALLOWED',
      confidence: 0.88,
      processingTime: 32,
      timestamp: '2024-01-15 14:25:10',
      modelVersions: {
        xgboost: '1.0.0',
        lstm: '1.0.0',
        autoencoder: '1.0.0',
        gnn: '1.0.0'
      },
      explanation: {
        risk_factors: [
          { feature: 'amount', impact: -0.1, direction: 'reduces', severity: 'low' },
          { feature: 'merchant_category', impact: -0.05, direction: 'reduces', severity: 'low' }
        ],
        human_readable: 'Low transaction amount (₹2,500) reduces fraud risk. Food merchant category reduces fraud risk.'
      },
      alerts: []
    },
    {
      key: '3',
      transactionId: 'TXN123456791',
      upiId: 'user@paytm',
      amount: 50000,
      merchant: 'Crypto Exchange',
      merchantCategory: 'crypto',
      deviceId: 'device_789',
      ipAddress: '192.168.1.102',
      location: { lat: 28.6139, lon: 77.2090 },
      riskScore: 0.92,
      fraudProbability: 0.94,
      decision: 'CHALLENGED',
      confidence: 0.95,
      processingTime: 67,
      timestamp: '2024-01-15 14:20:45',
      modelVersions: {
        xgboost: '1.0.0',
        lstm: '1.0.0',
        autoencoder: '1.0.0',
        gnn: '1.0.0'
      },
      explanation: {
        risk_factors: [
          { feature: 'amount', impact: 0.4, direction: 'increases', severity: 'high' },
          { feature: 'merchant_category', impact: 0.35, direction: 'increases', severity: 'high' },
          { feature: 'time_pattern', impact: 0.2, direction: 'increases', severity: 'medium' }
        ],
        human_readable: 'High transaction amount (₹50,000) increases fraud risk. High-risk merchant category (crypto) increases fraud risk. Unusual time pattern increases fraud risk.'
      },
      alerts: ['High amount transaction', 'High risk merchant', 'Unusual time pattern']
    }
  ];

  const columns = [
    {
      title: 'Transaction ID',
      dataIndex: 'transactionId',
      key: 'transactionId',
      width: 150,
    },
    {
      title: 'Amount',
      dataIndex: 'amount',
      key: 'amount',
      render: (amount) => `₹${amount.toLocaleString()}`,
      sorter: (a, b) => a.amount - b.amount,
    },
    {
      title: 'Merchant',
      dataIndex: 'merchant',
      key: 'merchant',
    },
    {
      title: 'Category',
      dataIndex: 'merchantCategory',
      key: 'merchantCategory',
      render: (category) => <Tag color="blue">{category}</Tag>,
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
      sorter: (a, b) => a.riskScore - b.riskScore,
    },
    {
      title: 'Decision',
      dataIndex: 'decision',
      key: 'decision',
      render: (decision) => (
        <Tag color={decision === 'BLOCKED' ? 'red' : decision === 'CHALLENGED' ? 'orange' : 'green'}>
          {decision}
        </Tag>
      ),
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence) => `${(confidence * 100).toFixed(1)}%`,
    },
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedTransaction(record);
            setIsModalVisible(true);
          }}
        >
          View Details
        </Button>
      ),
    },
  ];

  const handleViewDetails = (record) => {
    setSelectedTransaction(record);
    setIsModalVisible(true);
  };

  return (
    <div>
      <Card>
        <Space style={{ marginBottom: 16, width: '100%' }} direction="vertical" size="middle">
          <Space wrap>
            <Search
              placeholder="Search transactions..."
              style={{ width: 300 }}
              onSearch={(value) => console.log('Search:', value)}
            />
            <Select placeholder="Decision" style={{ width: 120 }}>
              <Option value="ALLOWED">Allowed</Option>
              <Option value="CHALLENGED">Challenged</Option>
              <Option value="BLOCKED">Blocked</Option>
            </Select>
            <Select placeholder="Risk Level" style={{ width: 120 }}>
              <Option value="high">High Risk</Option>
              <Option value="medium">Medium Risk</Option>
              <Option value="low">Low Risk</Option>
            </Select>
            <RangePicker />
            <Button icon={<FilterOutlined />}>Apply Filters</Button>
          </Space>
        </Space>

        <Table
          columns={columns}
          dataSource={transactions}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} transactions`,
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      <Modal
        title="Transaction Details"
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedTransaction && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Transaction ID" span={2}>
                {selectedTransaction.transactionId}
              </Descriptions.Item>
              <Descriptions.Item label="Amount">
                ₹{selectedTransaction.amount.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Merchant">
                {selectedTransaction.merchant}
              </Descriptions.Item>
              <Descriptions.Item label="UPI ID">
                {selectedTransaction.upiId}
              </Descriptions.Item>
              <Descriptions.Item label="Device ID">
                {selectedTransaction.deviceId}
              </Descriptions.Item>
              <Descriptions.Item label="IP Address">
                {selectedTransaction.ipAddress}
              </Descriptions.Item>
              <Descriptions.Item label="Location">
                {selectedTransaction.location.lat}, {selectedTransaction.location.lon}
              </Descriptions.Item>
              <Descriptions.Item label="Risk Score">
                <span className={selectedTransaction.riskScore > 0.7 ? 'risk-score-high' : selectedTransaction.riskScore > 0.4 ? 'risk-score-medium' : 'risk-score-low'}>
                  {(selectedTransaction.riskScore * 100).toFixed(1)}%
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="Decision">
                <Tag color={selectedTransaction.decision === 'BLOCKED' ? 'red' : selectedTransaction.decision === 'CHALLENGED' ? 'orange' : 'green'}>
                  {selectedTransaction.decision}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Confidence">
                {(selectedTransaction.confidence * 100).toFixed(1)}%
              </Descriptions.Item>
              <Descriptions.Item label="Processing Time">
                {selectedTransaction.processingTime}ms
              </Descriptions.Item>
              <Descriptions.Item label="Timestamp">
                {selectedTransaction.timestamp}
              </Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: 16 }}>
              <h4>Risk Factors</h4>
              {selectedTransaction.explanation.risk_factors.map((factor, index) => (
                <Tag key={index} color={factor.severity === 'high' ? 'red' : factor.severity === 'medium' ? 'orange' : 'green'}>
                  {factor.feature}: {factor.direction} risk ({factor.severity})
                </Tag>
              ))}
            </div>

            <div style={{ marginTop: 16 }}>
              <h4>Explanation</h4>
              <p>{selectedTransaction.explanation.human_readable}</p>
            </div>

            <div style={{ marginTop: 16 }}>
              <h4>Alerts</h4>
              {selectedTransaction.alerts.length > 0 ? (
                selectedTransaction.alerts.map((alert, index) => (
                  <Tag key={index} color="red">{alert}</Tag>
                ))
              ) : (
                <Tag color="green">No alerts</Tag>
              )}
            </div>

            <div style={{ marginTop: 16 }}>
              <h4>Model Versions</h4>
              <ReactJson src={selectedTransaction.modelVersions} theme="monokai" />
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Transactions;
