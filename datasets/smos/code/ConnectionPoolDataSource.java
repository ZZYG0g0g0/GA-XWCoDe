/*
 * ConnectionPoolDataSource
 *
 */

package smos.storage.connectionManagement;

import smos.storage.connectionManagement.exception.NotImplementedYetException;

import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.Driver;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.List;
import java.util.Properties;
import java.util.Vector;
import javax.sql.DataSource;

/**
 * Realization of the connection pool through the implementation
 * of the java interface. sql. DataSource. The pool periodically monitors the
 * active connections and those that are pools, i.e. those released but still
 * usable (present i.e. in memory). The release time of the
 * active connections and those pools are represented by two parameters present
 * within the class and which are connectionPoolCloseTimeout and
 * inactiveMaxTimeout; such values as well as all the others relating to the pool
 * have their default value, parameterizable via the file of
 * properties connection.properties
 */

public class ConnectionPoolDataSource implements DataSource {

    /**
     * Thread inside of the ConnectionPoolDataSource class that establishes each
     * connectionPoolCloseTimeout milliseconds the release of connections
     * pool.
     */
    private class ConnectionCloser extends Thread {

        private long connectionActionTimestamp = 0;

        private int connectionPoolCloseTimeout = 300000;

        private long timeStamp = 0;

        /**
         * Builder setting the release time of pool connections
         * 
         * @author Di Giorgio Domenico, Cris Malinconico
         * @param pTime
         *            time interval within which the pool empties the list
         *            pool connections.
         */
		private ConnectionCloser(int pTime) {
            setDaemon(true);
            setName("ConnectionPoolCloser");
            if (pTime > 0)
            	this.connectionPoolCloseTimeout = pTime;
        }

        /**
         * Each time a connection generates an event with an invocation of
         * getConnection() or release() timestamp is set to value
         * current time using this method.
         */
		public void connectionEvent() {
			this.connectionActionTimestamp = System.currentTimeMillis();
        }

        /**
         * Check every connectionPoolCloseTimeout milliseconds if the
         * pool connections can be closed, thus freeing up memory.
         */
		public void run() {
            boolean working = true;
            while (working) {
                try {
                	this.timeStamp = System.currentTimeMillis();
                    Thread.sleep(this.connectionPoolCloseTimeout);
                    if (this.connectionActionTimestamp < this.timeStamp) {
                        closeAllConnections(ConnectionPoolDataSource.this.pool);
                    }
                } catch (InterruptedException e) {
                    working = false;
                    e.printStackTrace();
                } catch (SQLException e) {
                    working = false;
                    e.printStackTrace();
                }
            }
        }
    }

    private List<SMOSConnection> active = new Vector<SMOSConnection>();

    private Properties config = new Properties();

    private ConnectionCloser connectionCloser;

    private Driver driver;

    private String fullConnectionString;

    private long inactiveMaxTimeout = 20000;

    private int maxPoolSize;

    private List<Connection> pool = new Vector<Connection>();

    /**
     * Creates a new instance of the connection pool.
     * 
     * @param pJdbcDriverName
     *            nome del driver jdbc
     * @param pFullConnectionString
     *            database connection string
     * @param pUser
     *            username (database administrator)
     * @param pPassword
     *            password amministratore
     * @param pMaxPoolSize
     *            administrator maximum number of active connections in the pool, must be
     *            greater than 0
     * @param pPoolTime
     *            time interval within which the pool will be emptied each
     *            time of its pool connections (in ms).
     * @throws ClassNotFoundException
     *             if jdbc driver cannot be found
     * @throws SQLException
     *             if there is a problem connecting to the database
     * @throws IllegalArgumentException
     *             if input parameters are invalid
     */

    
	public ConnectionPoolDataSource(String pJdbcDriverName,
            String pFullConnectionString, String pUser, String pPassword,
            int pMaxPoolSize, int pPoolTime) throws ClassNotFoundException,
            SQLException {

        if (pMaxPoolSize < 1) {
            throw new IllegalArgumentException(
                    "maxPoolSize must be >0 but it is: " + pMaxPoolSize);
        }
        if (pFullConnectionString == null) {
            throw new IllegalArgumentException("fullConnectionString "
                    + "has Null value");
        }
        if (pUser == null) {
            throw new IllegalArgumentException("Username is Null");
        }
        this.maxPoolSize = pMaxPoolSize;
        this.fullConnectionString = pFullConnectionString;
        this.config.put("user", pUser);
        if (pPassword != null) {
            this.config.put("password", pPassword);
        }
        Class.forName(pJdbcDriverName);
        this.driver = DriverManager.getDriver(pFullConnectionString);
        this.connectionCloser = new ConnectionCloser(pPoolTime);
        this.connectionCloser.start();
    }

    /**
     * Returns the size of the list of active connections.
     * 
     * @return the size of the list of currently active connections.
     */
	public int activeSize() {
        return this.active.size();
    }

    /**
     * Empty the pool of connections from active ones that have no longer run
     * operations for inactiveMaxTimeout milliseconds.
     * 
     */
    protected void clearActive() {
        long temp = 0;
        long TIME = System.currentTimeMillis();
        SMOSConnection adc = null;

        for (int count = 0; count < this.active.size(); count++) {
            adc = (SMOSConnection) this.active.get(count);
            temp = TIME - adc.getLastTime();
            if (temp >= this.inactiveMaxTimeout) {
                this.release(adc.getConnection());
            }
        }
    }

    /**
     * Closes all connections in the pool both active and
     * They're pools.
     * 
     * @author Di Giorgio Domenico, Cris Malinconico
     * @throws SQLException
     */
    public synchronized void closeAllConnections() throws SQLException {
        closeAllConnections(this.pool);
        closeAllConnections(this.active);
    }

    /**
     * Closes all connections indicated in the connection list.
     * 
     * @author Di Giorgio Domenico, Cris Malinconico
     * @param pConnections
     *            the list of connections to be closed.
     * @throws SQLException
     *             when it is impossible to close a connection.
     */
    private synchronized void closeAllConnections(List pConnections)
            throws SQLException {

        while (pConnections.size() > 0) {
            ConnectionWrapper conn = (ConnectionWrapper) pConnections.remove(0);
            conn.closeWrappedConnection();
        }
    }

    /**
     * Closes all pool connections that are in the pool list.
     * 
     * @throws SQLException
     *             when it is impossible to close a connection.
     */
    public synchronized void closeAllPooledConnections() throws SQLException {
        closeAllConnections(this.pool);
    }

    /**
     * Method used by getConnection() to create a new connection
     * if they are not in the pool list.
     * 
     * @return a new connection to the DataBase.
     */
    private synchronized Connection createNewConnection() {
        Connection rawConn = null;
        try {
            rawConn = this.driver.connect(this.fullConnectionString, this.config);
            Connection conn = new ConnectionWrapper(rawConn, this);
            SMOSConnection ac = new SMOSConnection();
            ac.setConnection(conn);
            ac.setLastTime(System.currentTimeMillis());
            this.active.add(ac);
            return conn;
        } catch (SQLException e) {
            System.out.println("Connection creation failed "
                    + "in ConnectionPoolDataSource:" + e);
            return null;
        }
    }

    /**
     * Returns a connection if the pool is not full, the check is done
     * first in the list of pool connections to avoid creations
     * useless otherwise a new connection will be created.
     * 
     * @return the connection to the database if possible otherwise
     *         an exception is generated
     * @see javax.sql.DataSource getConnection()
     * @throws SQLException
     *             If a problem occurs while connecting to the database
     *                      including the fact that the maximum limit of connections
     *                      active is reached.
     */
    public synchronized Connection getConnection() throws SQLException {

        Connection connection = getPooledConnection(0);

        if (connection == null) {
            if (this.active.size() >= this.maxPoolSize) {
                throw new SQLException("Connection pool limit of "
                        + this.maxPoolSize + " exceeded");
            } else {
                connection = createNewConnection();
            }
        }
        this.connectionCloser.connectionEvent();
        //System.out.println("GET CONNECTION: " + active.size() + "/" + pool.size());
        return connection;
    }

    /**
     * Method not implemented
     * @param pArg1 
     * @param pArg2 
     * @return Connection
     * @throws SQLException 
     * 
     * @throws NotImplementedYetException
     */

    public Connection getConnection(String pArg1, String pArg2)
            throws SQLException {
        throw new NotImplementedYetException();
    }

    /**
     * Method not implemented
     * @return int
     * @throws SQLException 
     * 
     * @throws NotImplementedYetException
     */

    public int getLoginTimeout() throws SQLException {
        throw new NotImplementedYetException();
    }

    /**
     * Method not implemented
     * @return PrintWriter
     * @throws SQLException 
     * 
     * @throws NotImplementedYetException
     */

    public PrintWriter getLogWriter() throws SQLException {
        throw new NotImplementedYetException();
    }

    /**
     * Returns the maximum number of active connections
     * 
     * @return the maximum number of active connections.
     */

    public int getMaxPoolSize() {
        return this.maxPoolSize;
    }

    /**
     * Method used by getConnection() to determine whether in the list of
     * pool connections there are some that you can reuse.
     * 
     * @param pPoolIndex
     *            index of the pool connection list (always 0).
     * @return a connection from the list of those pools if any
     *         one.
     */
    private synchronized Connection getPooledConnection(int pPoolIndex) {
        SMOSConnection ac = new SMOSConnection();
        Connection connection = null;
        if (this.pool.size() > 0) {
            connection = (Connection) this.pool.remove(pPoolIndex);
            ac.setConnection(connection);
            ac.setLastTime(System.currentTimeMillis());
            this.active.add(ac);
        }
        return ac.getConnection();
    }

    /**
     * Returns the size of the pool connection list
     * 
     * @return the size of the pool connection list.
     */
    public int poolSize() {
        return this.pool.size();
    }

    /**
     * Release a connection, deleting it from active ones and inserting it into
     * those pools to be reused later.
     * 
     * @param pConnection
     *            The connection that must be returned to the pool.
     */
    public synchronized void release(Connection pConnection) {
        boolean exists = false;
        int activeIndex = 0;

        if (pConnection != null) {
            SMOSConnection adc = null;
            while ((activeIndex < this.active.size()) && (!exists)) {
                adc = (SMOSConnection) this.active.get(activeIndex);
                if (adc.equals(pConnection)) {
                	this.active.remove(adc);
                	this.pool.add(adc.getConnection());
                    exists = true;
                }
                activeIndex++;
            }
            this.connectionCloser.connectionEvent();
            //System.out.println("RELEASE CONNECTION: " + active.size() + "/" + pool.size());
        }
    }

    /**
     * Sets the lifetime of active connections in milliseconds.
     * 
     * @param pTimeOut
     *            connection life time.
     */

    public void setActivedTimeout(long pTimeOut) {
        if (pTimeOut > 0) {
        	this.inactiveMaxTimeout = pTimeOut;
        }
    }

    /**
     * Rearrange the lifetime of the connection due to run
     * an operation. From now on the connection can be active without
     * perform any operation for other inactiveMaxTimeout milliseconds.
     * 
     * @param pConnection
     *            the connection that performed an operation and therefore can
     *            Stay active.
     */

    void setLastTime(Connection pConnection) {
        boolean exists = false;
        int count = 0;
        SMOSConnection adc = null;

        while ((count < this.active.size()) && (!exists)) {
            adc = (SMOSConnection) this.active.get(count);
            count++;
            if (adc.equals(pConnection)) {
                adc.setLastTime(System.currentTimeMillis());
                exists = true;
            }
        }
    }

    /**
     * Method not implemented
     * @param pArg0 
     * @throws SQLException 
     * 
     * @throws NotImplementedYetException
     */
    public void setLoginTimeout(int pArg0) throws SQLException {
        throw new NotImplementedYetException();
    }

    /**
     * Method not implemented
     * @param pArg0 
     * @throws SQLException 
     * 
     * @throws NotImplementedYetException
     */
    public void setLogWriter(PrintWriter pArg0) throws SQLException {
        throw new NotImplementedYetException();
    }

    /**
     * Converts an object of the ConnectionPoolDataSource class to String
     * 
     * @return the String representation of the connection pool.
     */

    public String toString() {
        StringBuffer buf = new StringBuffer();

        buf.append("[");
        buf.append("maxPoolSize=").append(this.maxPoolSize);
        buf.append(", activeSize=").append(activeSize());
        buf.append(", poolSize=").append(poolSize());
        buf.append(", fullConnectionString=").append(this.fullConnectionString);
        buf.append("]");
        return (buf.toString());
    }

	
}
