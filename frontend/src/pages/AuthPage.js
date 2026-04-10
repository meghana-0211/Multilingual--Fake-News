import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import styles from './AuthPage.module.css';

function AuthForm({ mode }) {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail]       = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  const isLogin = mode === 'login';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(email, username, password);
      }
      navigate('/');
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        {/* Logo */}
        <div className={styles.logoRow}>
          <div className={styles.logoMark}>S</div>
          <span className={styles.logoText}>SatyaCheck</span>
        </div>

        <h1 className={styles.title}>{isLogin ? 'Sign in' : 'Create account'}</h1>
        <p className={styles.sub}>
          {isLogin
            ? 'Access history, feedback, and admin tools.'
            : 'Join the fact-checking network.'}
        </p>

        <form className={styles.form} onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoFocus
              className={styles.input}
            />
          </div>

          {!isLogin && (
            <div className={styles.field}>
              <label>Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="yourname"
                required
                className={styles.input}
              />
            </div>
          )}

          <div className={styles.field}>
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder={isLogin ? '••••••••' : 'At least 8 characters'}
              required
              className={styles.input}
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button type="submit" className={styles.submitBtn} disabled={loading}>
            {loading ? 'Please wait…' : (isLogin ? 'Sign in' : 'Create account')}
          </button>
        </form>

        <p className={styles.switchText}>
          {isLogin ? "Don't have an account? " : 'Already have an account? '}
          <Link to={isLogin ? '/register' : '/login'} className={styles.switchLink}>
            {isLogin ? 'Register' : 'Sign in'}
          </Link>
        </p>
      </div>
    </div>
  );
}

export function LoginPage()    { return <AuthForm mode="login"    />; }
export function RegisterPage() { return <AuthForm mode="register" />; }