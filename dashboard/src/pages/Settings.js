import React from 'react';
import { Card, Form, Input, Button, Switch, Select, Slider, InputNumber, Divider, message } from 'antd';
import { SaveOutlined, ReloadOutlined } from '@ant-design/icons';

const { Option } = Select;
const { TextArea } = Input;

const Settings = () => {
  const [form] = Form.useForm();

  const onFinish = (values) => {
    console.log('Settings saved:', values);
    message.success('Settings saved successfully!');
  };

  const handleReset = () => {
    form.resetFields();
    message.info('Settings reset to default values');
  };

  return (
    <div>
      <Card title="System Settings" style={{ marginBottom: 16 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={onFinish}
          initialValues={{
            systemName: 'UPI Fraud Detection System',
            version: '1.0.0',
            debugMode: false,
            logLevel: 'INFO',
            maxConnections: 1000,
            requestTimeout: 30,
            enableMonitoring: true,
            enableAlerts: true,
            alertEmail: 'admin@company.com',
            riskThresholdHigh: 0.8,
            riskThresholdMedium: 0.5,
            riskThresholdLow: 0.2,
            modelRetrainInterval: 24,
            featureCacheTTL: 3600,
            maxRetries: 3,
            enableRateLimit: true,
            rateLimitPerMinute: 1000,
          }}
        >
          <Row gutter={[24, 24]}>
            <Col xs={24} lg={12}>
              <Card title="General Settings" size="small">
                <Form.Item
                  label="System Name"
                  name="systemName"
                  rules={[{ required: true, message: 'Please input system name!' }]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  label="Version"
                  name="version"
                  rules={[{ required: true, message: 'Please input version!' }]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  label="Debug Mode"
                  name="debugMode"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>

                <Form.Item
                  label="Log Level"
                  name="logLevel"
                >
                  <Select>
                    <Option value="DEBUG">DEBUG</Option>
                    <Option value="INFO">INFO</Option>
                    <Option value="WARNING">WARNING</Option>
                    <Option value="ERROR">ERROR</Option>
                  </Select>
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Performance Settings" size="small">
                <Form.Item
                  label="Max Connections"
                  name="maxConnections"
                  rules={[{ required: true, message: 'Please input max connections!' }]}
                >
                  <InputNumber min={1} max={10000} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item
                  label="Request Timeout (seconds)"
                  name="requestTimeout"
                  rules={[{ required: true, message: 'Please input request timeout!' }]}
                >
                  <InputNumber min={1} max={300} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item
                  label="Feature Cache TTL (seconds)"
                  name="featureCacheTTL"
                  rules={[{ required: true, message: 'Please input cache TTL!' }]}
                >
                  <InputNumber min={60} max={86400} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item
                  label="Max Retries"
                  name="maxRetries"
                  rules={[{ required: true, message: 'Please input max retries!' }]}
                >
                  <InputNumber min={0} max={10} style={{ width: '100%' }} />
                </Form.Item>
              </Card>
            </Col>
          </Row>

          <Divider />

          <Row gutter={[24, 24]}>
            <Col xs={24} lg={12}>
              <Card title="Risk Thresholds" size="small">
                <Form.Item
                  label="High Risk Threshold"
                  name="riskThresholdHigh"
                  rules={[{ required: true, message: 'Please set high risk threshold!' }]}
                >
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    marks={{
                      0: '0',
                      0.5: '0.5',
                      1: '1'
                    }}
                  />
                </Form.Item>

                <Form.Item
                  label="Medium Risk Threshold"
                  name="riskThresholdMedium"
                  rules={[{ required: true, message: 'Please set medium risk threshold!' }]}
                >
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    marks={{
                      0: '0',
                      0.5: '0.5',
                      1: '1'
                    }}
                  />
                </Form.Item>

                <Form.Item
                  label="Low Risk Threshold"
                  name="riskThresholdLow"
                  rules={[{ required: true, message: 'Please set low risk threshold!' }]}
                >
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    marks={{
                      0: '0',
                      0.5: '0.5',
                      1: '1'
                    }}
                  />
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Model Settings" size="small">
                <Form.Item
                  label="Model Retrain Interval (hours)"
                  name="modelRetrainInterval"
                  rules={[{ required: true, message: 'Please input retrain interval!' }]}
                >
                  <InputNumber min={1} max={168} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item
                  label="Enable Monitoring"
                  name="enableMonitoring"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>

                <Form.Item
                  label="Enable Alerts"
                  name="enableAlerts"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>

                <Form.Item
                  label="Alert Email"
                  name="alertEmail"
                  rules={[
                    { required: true, message: 'Please input alert email!' },
                    { type: 'email', message: 'Please input valid email!' }
                  ]}
                >
                  <Input />
                </Form.Item>
              </Card>
            </Col>
          </Row>

          <Divider />

          <Row gutter={[24, 24]}>
            <Col xs={24} lg={12}>
              <Card title="Rate Limiting" size="small">
                <Form.Item
                  label="Enable Rate Limiting"
                  name="enableRateLimit"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>

                <Form.Item
                  label="Rate Limit (requests per minute)"
                  name="rateLimitPerMinute"
                  rules={[{ required: true, message: 'Please input rate limit!' }]}
                >
                  <InputNumber min={1} max={10000} style={{ width: '100%' }} />
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Database Settings" size="small">
                <Form.Item
                  label="PostgreSQL Host"
                  name="postgresHost"
                  rules={[{ required: true, message: 'Please input PostgreSQL host!' }]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  label="PostgreSQL Port"
                  name="postgresPort"
                  rules={[{ required: true, message: 'Please input PostgreSQL port!' }]}
                >
                  <InputNumber min={1} max={65535} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item
                  label="Redis Host"
                  name="redisHost"
                  rules={[{ required: true, message: 'Please input Redis host!' }]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  label="Redis Port"
                  name="redisPort"
                  rules={[{ required: true, message: 'Please input Redis port!' }]}
                >
                  <InputNumber min={1} max={65535} style={{ width: '100%' }} />
                </Form.Item>
              </Card>
            </Col>
          </Row>

          <Divider />

          <Card title="Advanced Settings" size="small">
            <Form.Item
              label="Custom Configuration"
              name="customConfig"
            >
              <TextArea
                rows={6}
                placeholder="Enter custom JSON configuration..."
              />
            </Form.Item>
          </Card>

          <div style={{ textAlign: 'center', marginTop: 24 }}>
            <Space size="large">
              <Button
                type="primary"
                htmlType="submit"
                icon={<SaveOutlined />}
                size="large"
              >
                Save Settings
              </Button>
              <Button
                onClick={handleReset}
                icon={<ReloadOutlined />}
                size="large"
              >
                Reset to Default
              </Button>
            </Space>
          </div>
        </Form>
      </Card>
    </div>
  );
};

export default Settings;
