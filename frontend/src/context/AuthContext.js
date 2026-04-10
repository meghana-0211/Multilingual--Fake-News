import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadUser = useCallback(async () => {
    const token = localStorage.getItem('token');
    if (!token) { setLoading(false); return; }
    try {
      const data = await api.me();
      setUser(data.user);
    } catch {
      localStorage.removeItem('token');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadUser(); }, [loadUser]);

  const login = async (email, password) => {
    const data = await api.login(email, password);
    localStorage.setItem('token', data.accessToken);
    setUser(data.user);
    return data.user;
  };

  const register = async (email, username, password) => {
    const data = await api.register(email, username, password);
    localStorage.setItem('token', data.accessToken);
    setUser(data.user);
    return data.user;
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  const isAdmin = user?.role === 'admin';
  const isFactChecker = user?.role === 'fact_checker' || isAdmin;
  const isPublisher = user?.role === 'publisher' || isAdmin;

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, isAdmin, isFactChecker, isPublisher }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);