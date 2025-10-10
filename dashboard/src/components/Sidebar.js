import React from 'react';
import { Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  TransactionOutlined,
  BarChartOutlined,
  FileTextOutlined,
  RobotOutlined,
  SettingOutlined
} from '@ant-design/icons';

const Sidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/transactions',
      icon: <TransactionOutlined />,
      label: 'Transactions',
    },
    {
      key: '/analytics',
      icon: <BarChartOutlined />,
      label: 'Analytics',
    },
    {
      key: '/cases',
      icon: <FileTextOutlined />,
      label: 'Fraud Cases',
    },
    {
      key: '/models',
      icon: <RobotOutlined />,
      label: 'Models',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
  ];

  return (
    <div style={{ width: 200, background: '#001529', height: '100vh' }}>
      <div style={{ 
        padding: '16px', 
        color: 'white', 
        fontSize: '18px', 
        fontWeight: 'bold',
        borderBottom: '1px solid #1890ff'
      }}>
        UPI Fraud Detection
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
        style={{ border: 'none' }}
      />
    </div>
  );
};

export default Sidebar;
