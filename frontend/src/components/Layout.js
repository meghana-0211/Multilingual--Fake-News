import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import styles from './Layout.module.css';

const NAV_LINKS = [
  { to: '/',         label: 'Analyse',  icon: '⬡' },
  { to: '/history',  label: 'History',  icon: '⬡' },
  { to: '/feedback', label: 'Feedback', icon: '⬡' },
  { to: '/admin',    label: 'Admin',    icon: '⬡', adminOnly: true },
];

export default function Layout({ children }) {
  const { user, logout, isAdmin } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  const handleLogout = () => { logout(); navigate('/login'); };

  const links = NAV_LINKS.filter(l => !l.adminOnly || isAdmin);

  return (
    <div className={styles.shell}>
      {/* Top nav */}
      <header className={styles.header}>
        <Link to="/" className={styles.logo}>
          <span className={styles.logoMark}>S</span>
          <span className={styles.logoText}>SatyaCheck</span>
          <span className={styles.logoBadge}>BETA</span>
        </Link>

        <nav className={`${styles.nav} ${menuOpen ? styles.navOpen : ''}`}>
          {links.map(l => (
            <Link
              key={l.to}
              to={l.to}
              className={`${styles.navLink} ${location.pathname === l.to ? styles.active : ''}`}
              onClick={() => setMenuOpen(false)}
            >
              {l.label}
            </Link>
          ))}
        </nav>

        <div className={styles.headerRight}>
          {user ? (
            <>
              <span className={styles.userBadge}>{user.role}</span>
              <span className={styles.userName}>{user.username}</span>
              <button className={styles.logoutBtn} onClick={handleLogout}>Sign out</button>
            </>
          ) : (
            <>
              <Link to="/login"    className={styles.authLink}>Sign in</Link>
              <Link to="/register" className={styles.authBtn}>Register</Link>
            </>
          )}
          <button className={styles.burger} onClick={() => setMenuOpen(v => !v)}>
            <span /><span /><span />
          </button>
        </div>
      </header>

      {/* Page content */}
      <main className={styles.main}>{children}</main>

      {/* Footer */}
      <footer className={styles.footer}>
        <span className={styles.mono}>SatyaCheck v1.0 — IndicBERT + BiLSTM + Blockchain</span>
        <span className={styles.footerLangs}>हिं · ગુ · मरा · తె</span>
      </footer>
    </div>
  );
}