import React from 'react';
import { Layout, Button, Space, Badge, Avatar, Dropdown, Menu } from 'antd';
import { BellOutlined, UserOutlined, LogoutOutlined, ReloadOutlined } from '@ant-design/icons';

const { Header: AntHeader } = Layout;

const Header = () => {
  const handleRefresh = () => {
    window.location.reload();
  };

  const userMenu = (
    <Menu>
      <Menu.Item key="profile" icon={<UserOutlined />}>
        Profile
      </Menu.Item>
      <Menu.Item key="logout" icon={<LogoutOutlined />}>
        Logout
      </Menu.Item>
    </Menu>
  );

  return (
    <AntHeader style={{ 
      padding: '0 24px', 
      background: '#fff',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
    }}>
      <div>
        <h2 style={{ margin: 0, color: '#1890ff' }}>Real-time Fraud Detection</h2>
      </div>
      
      <Space size="middle">
        <Button 
          icon={<ReloadOutlined />} 
          onClick={handleRefresh}
          type="text"
        >
          Refresh
        </Button>
        
        <Badge count={5} size="small">
          <Button 
            icon={<BellOutlined />} 
            type="text"
            size="large"
          />
        </Badge>
        
        <Dropdown overlay={userMenu} placement="bottomRight">
          <Avatar 
            icon={<UserOutlined />} 
            style={{ cursor: 'pointer' }}
          />
        </Dropdown>
      </Space>
    </AntHeader>
  );
};

export default Header;
