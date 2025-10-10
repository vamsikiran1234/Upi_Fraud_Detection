import React from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Tag, Button, Space, Modal, Descriptions } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Models = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  // Mock data for models
  const models = [
    {
      key: '1',
      name: 'XGBoost',
      version: '1.0.0',
      status: 'active',
      accuracy: 94.5,
      precision: 92.1,
      recall: 89.3,
      f1Score: 90.7,
      lastTrained: '2024-01-15 10:30:00',
      trainingTime: '2h 15m',
      predictions: 12543,
      errors: 12,
      latency: 45,
    },
    {
      key: '2',
      name: 'LSTM',
      version: '1.0.0',
      status: 'active',
      accuracy: 91.2,
      precision: 88.5,
      recall: 92.1,
      f1Score: 90.3,
      lastTrained: '2024-01-15 08:45:00',
      trainingTime: '4h 30m',
      predictions: 12543,
      errors: 8,
      latency: 67,
    },
    {
      key: '3',
      name: 'Autoencoder',
      version: '1.0.0',
      status: 'active',
      accuracy: 89.8,
      precision: 85.2,
      recall: 94.5,
      f1Score: 89.6,
      lastTrained: '2024-01-15 06:20:00',
      trainingTime: '3h 45m',
      predictions: 12543,
      errors: 15,
      latency: 52,
    },
    {
      key: '4',
      name: 'GNN',
      version: '1.0.0',
      status: 'training',
      accuracy: 92.1,
      precision: 90.3,
      recall: 87.8,
      f1Score: 89.0,
      lastTrained: '2024-01-15 12:00:00',
      trainingTime: '6h 20m',
      predictions: 12543,
      errors: 5,
      latency: 89,
    },
    {
      key: '5',
      name: 'Ensemble',
      version: '1.0.0',
      status: 'active',
      accuracy: 96.8,
      precision: 95.2,
      recall: 93.7,
      f1Score: 94.4,
      lastTrained: '2024-01-15 14:15:00',
      trainingTime: '1h 30m',
      predictions: 12543,
      errors: 3,
      latency: 78,
    },
  ];

  const performanceData = [
    { date: '2024-01-01', accuracy: 94.2, precision: 91.8, recall: 88.5 },
    { date: '2024-01-02', accuracy: 94.5, precision: 92.1, recall: 89.1 },
    { date: '2024-01-03', accuracy: 94.8, precision: 92.3, recall: 89.5 },
    { date: '2024-01-04', accuracy: 95.1, precision: 92.8, recall: 90.2 },
    { date: '2024-01-05', accuracy: 95.4, precision: 93.2, recall: 90.8 },
    { date: '2024-01-06', accuracy: 95.7, precision: 93.5, recall: 91.2 },
    { date: '2024-01-07', accuracy: 96.0, precision: 94.0, recall: 91.8 },
  ];

  const columns = [
    {
      title: 'Model Name',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <strong>{name}</strong>,
    },
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const colors = {
          active: 'green',
          training: 'blue',
          inactive: 'gray',
          error: 'red',
        };
        return <Tag color={colors[status]}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy) => (
        <div>
          <Progress
            percent={accuracy}
            size="small"
            status={accuracy > 95 ? 'success' : accuracy > 90 ? 'normal' : 'exception'}
          />
          <span style={{ marginLeft: 8 }}>{accuracy}%</span>
        </div>
      ),
    },
    {
      title: 'F1 Score',
      dataIndex: 'f1Score',
      key: 'f1Score',
      render: (score) => `${score}%`,
    },
    {
      title: 'Latency (ms)',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency) => (
        <span className={latency < 50 ? 'risk-score-low' : latency < 100 ? 'risk-score-medium' : 'risk-score-high'}>
          {latency}ms
        </span>
      ),
    },
    {
      title: 'Predictions',
      dataIndex: 'predictions',
      key: 'predictions',
      render: (predictions) => predictions.toLocaleString(),
    },
    {
      title: 'Errors',
      dataIndex: 'errors',
      key: 'errors',
      render: (errors) => (
        <span className={errors < 5 ? 'risk-score-low' : errors < 10 ? 'risk-score-medium' : 'risk-score-high'}>
          {errors}
        </span>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedModel(record);
              setIsModalVisible(true);
            }}
          >
            Details
          </Button>
          <Button
            type="link"
            icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
          >
            {record.status === 'active' ? 'Pause' : 'Start'}
          </Button>
          <Button
            type="link"
            icon={<ReloadOutlined />}
            disabled={record.status === 'training'}
          >
            Retrain
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Active Models"
              value={4}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Training Models"
              value={1}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Average Accuracy"
              value={92.9}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Average Latency"
              value={66.2}
              precision={1}
              suffix="ms"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24}>
          <Card title="Model Performance Over Time" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="accuracy" stroke="#52c41a" strokeWidth={2} name="Accuracy %" />
                <Line type="monotone" dataKey="precision" stroke="#1890ff" strokeWidth={2} name="Precision %" />
                <Line type="monotone" dataKey="recall" stroke="#faad14" strokeWidth={2} name="Recall %" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Card title="Model Management">
        <Space style={{ marginBottom: 16 }}>
          <Button type="primary">Deploy New Model</Button>
          <Button>Bulk Retrain</Button>
          <Button>Export Models</Button>
        </Space>

        <Table
          columns={columns}
          dataSource={models}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} models`,
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      <Modal
        title="Model Details"
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsModalVisible(false)}>
            Close
          </Button>,
          <Button key="retrain" type="primary" icon={<ReloadOutlined />}>
            Retrain Model
          </Button>,
        ]}
        width={800}
      >
        {selectedModel && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Model Name" span={2}>
                {selectedModel.name}
              </Descriptions.Item>
              <Descriptions.Item label="Version">
                {selectedModel.version}
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={selectedModel.status === 'active' ? 'green' : selectedModel.status === 'training' ? 'blue' : 'gray'}>
                  {selectedModel.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Accuracy">
                <Progress percent={selectedModel.accuracy} size="small" />
                <span style={{ marginLeft: 8 }}>{selectedModel.accuracy}%</span>
              </Descriptions.Item>
              <Descriptions.Item label="Precision">
                {selectedModel.precision}%
              </Descriptions.Item>
              <Descriptions.Item label="Recall">
                {selectedModel.recall}%
              </Descriptions.Item>
              <Descriptions.Item label="F1 Score">
                {selectedModel.f1Score}%
              </Descriptions.Item>
              <Descriptions.Item label="Latency">
                <span className={selectedModel.latency < 50 ? 'risk-score-low' : selectedModel.latency < 100 ? 'risk-score-medium' : 'risk-score-high'}>
                  {selectedModel.latency}ms
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="Predictions">
                {selectedModel.predictions.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Errors">
                <span className={selectedModel.errors < 5 ? 'risk-score-low' : selectedModel.errors < 10 ? 'risk-score-medium' : 'risk-score-high'}>
                  {selectedModel.errors}
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="Last Trained">
                {selectedModel.lastTrained}
              </Descriptions.Item>
              <Descriptions.Item label="Training Time">
                {selectedModel.trainingTime}
              </Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: 16 }}>
              <h4>Model Configuration</h4>
              <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
                {JSON.stringify({
                  algorithm: selectedModel.name,
                  version: selectedModel.version,
                  parameters: {
                    learning_rate: 0.1,
                    max_depth: 6,
                    n_estimators: 100,
                    random_state: 42
                  },
                  features: 20,
                  training_samples: 100000,
                  validation_samples: 20000
                }, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Models;
