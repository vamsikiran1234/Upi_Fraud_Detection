import React, { useState } from 'react';
import { Table, Card, Tag, Button, Space, Modal, Descriptions, Steps, Select, Input } from 'antd';
import { EyeOutlined, EditOutlined, CheckOutlined, CloseOutlined } from '@ant-design/icons';

const { Option } = Select;
const { TextArea } = Input;

const Cases = () => {
  const [selectedCase, setSelectedCase] = useState(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  // Mock data for fraud cases
  const cases = [
    {
      key: '1',
      caseId: 'CASE-001',
      transactionId: 'TXN123456789',
      status: 'open',
      priority: 'high',
      assignedTo: 'John Doe',
      description: 'High amount transaction with suspicious device',
      riskScore: 0.92,
      amount: 150000,
      merchant: 'Crypto Exchange',
      createdAt: '2024-01-15 14:30:25',
      updatedAt: '2024-01-15 15:45:10',
      resolution: null,
      resolvedAt: null,
    },
    {
      key: '2',
      caseId: 'CASE-002',
      transactionId: 'TXN123456790',
      status: 'investigating',
      priority: 'medium',
      assignedTo: 'Jane Smith',
      description: 'Multiple failed login attempts from new device',
      riskScore: 0.78,
      amount: 50000,
      merchant: 'Amazon India',
      createdAt: '2024-01-15 13:20:15',
      updatedAt: '2024-01-15 14:15:30',
      resolution: null,
      resolvedAt: null,
    },
    {
      key: '3',
      caseId: 'CASE-003',
      transactionId: 'TXN123456791',
      status: 'resolved',
      priority: 'low',
      assignedTo: 'Mike Johnson',
      description: 'False positive - legitimate transaction',
      riskScore: 0.45,
      amount: 25000,
      merchant: 'Swiggy',
      createdAt: '2024-01-15 12:10:05',
      updatedAt: '2024-01-15 13:30:20',
      resolution: 'False positive - transaction approved after verification',
      resolvedAt: '2024-01-15 13:30:20',
    },
  ];

  const columns = [
    {
      title: 'Case ID',
      dataIndex: 'caseId',
      key: 'caseId',
      width: 100,
    },
    {
      title: 'Transaction ID',
      dataIndex: 'transactionId',
      key: 'transactionId',
      width: 150,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const colors = {
          open: 'red',
          investigating: 'orange',
          resolved: 'green',
          closed: 'gray',
        };
        return <Tag color={colors[status]}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority) => {
        const colors = {
          high: 'red',
          medium: 'orange',
          low: 'green',
        };
        return <Tag color={colors[priority]}>{priority.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Assigned To',
      dataIndex: 'assignedTo',
      key: 'assignedTo',
    },
    {
      title: 'Amount',
      dataIndex: 'amount',
      key: 'amount',
      render: (amount) => `₹${amount.toLocaleString()}`,
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
      title: 'Created At',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 150,
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
              setSelectedCase(record);
              setIsModalVisible(true);
            }}
          >
            View
          </Button>
          <Button
            type="link"
            icon={<EditOutlined />}
            disabled={record.status === 'resolved'}
          >
            Edit
          </Button>
        </Space>
      ),
    },
  ];

  const getStatusSteps = (status) => {
    const steps = [
      { title: 'Case Created', status: 'finish' },
      { title: 'Under Investigation', status: status === 'open' ? 'process' : 'finish' },
      { title: 'Resolution', status: status === 'resolved' ? 'finish' : 'wait' },
    ];
    return steps;
  };

  return (
    <div>
      <Card>
        <Space style={{ marginBottom: 16, width: '100%' }} direction="vertical" size="middle">
          <Space wrap>
            <Select placeholder="Status" style={{ width: 120 }}>
              <Option value="open">Open</Option>
              <Option value="investigating">Investigating</Option>
              <Option value="resolved">Resolved</Option>
            </Select>
            <Select placeholder="Priority" style={{ width: 120 }}>
              <Option value="high">High</Option>
              <Option value="medium">Medium</Option>
              <Option value="low">Low</Option>
            </Select>
            <Select placeholder="Assigned To" style={{ width: 150 }}>
              <Option value="john">John Doe</Option>
              <Option value="jane">Jane Smith</Option>
              <Option value="mike">Mike Johnson</Option>
            </Select>
            <Button type="primary">Create New Case</Button>
          </Space>
        </Space>

        <Table
          columns={columns}
          dataSource={cases}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} cases`,
          }}
          scroll={{ x: 1000 }}
        />
      </Card>

      <Modal
        title="Case Details"
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsModalVisible(false)}>
            Close
          </Button>,
          <Button key="edit" type="primary" icon={<EditOutlined />}>
            Edit Case
          </Button>,
        ]}
        width={800}
      >
        {selectedCase && (
          <div>
            <Steps
              current={selectedCase.status === 'open' ? 1 : selectedCase.status === 'investigating' ? 2 : 3}
              items={getStatusSteps(selectedCase.status)}
              style={{ marginBottom: 24 }}
            />

            <Descriptions bordered column={2}>
              <Descriptions.Item label="Case ID" span={2}>
                {selectedCase.caseId}
              </Descriptions.Item>
              <Descriptions.Item label="Transaction ID">
                {selectedCase.transactionId}
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={selectedCase.status === 'open' ? 'red' : selectedCase.status === 'investigating' ? 'orange' : 'green'}>
                  {selectedCase.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Priority">
                <Tag color={selectedCase.priority === 'high' ? 'red' : selectedCase.priority === 'medium' ? 'orange' : 'green'}>
                  {selectedCase.priority.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Assigned To">
                {selectedCase.assignedTo}
              </Descriptions.Item>
              <Descriptions.Item label="Amount">
                ₹{selectedCase.amount.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Merchant">
                {selectedCase.merchant}
              </Descriptions.Item>
              <Descriptions.Item label="Risk Score">
                <span className={selectedCase.riskScore > 0.7 ? 'risk-score-high' : selectedCase.riskScore > 0.4 ? 'risk-score-medium' : 'risk-score-low'}>
                  {(selectedCase.riskScore * 100).toFixed(1)}%
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="Created At">
                {selectedCase.createdAt}
              </Descriptions.Item>
              <Descriptions.Item label="Updated At">
                {selectedCase.updatedAt}
              </Descriptions.Item>
              {selectedCase.resolvedAt && (
                <Descriptions.Item label="Resolved At">
                  {selectedCase.resolvedAt}
                </Descriptions.Item>
              )}
            </Descriptions>

            <div style={{ marginTop: 16 }}>
              <h4>Description</h4>
              <p>{selectedCase.description}</p>
            </div>

            {selectedCase.resolution && (
              <div style={{ marginTop: 16 }}>
                <h4>Resolution</h4>
                <p>{selectedCase.resolution}</p>
              </div>
            )}

            <div style={{ marginTop: 16 }}>
              <h4>Add Comment</h4>
              <TextArea
                rows={4}
                placeholder="Add your comment here..."
                style={{ marginBottom: 8 }}
              />
              <Button type="primary" icon={<CheckOutlined />}>
                Add Comment
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Cases;
