package smos.storage.connectionManagement;


import java.sql.*;

/**
 * Implementation of the thread able to maintain the connection to the database
 * Mysql if there are no more active connections (troubleshooting)
 * Mysql autoreconnect ). This class is also responsible for recalling the
 * method of releasing active connections that have not performed operations
 * in a certain interval of time.
 */

public class ControlConnection extends Thread {
    private static ConnectionPoolDataSource manager = null;

    private static int waitTimeout;

    /**
     * Creates a new instance of the Thread.
     * 
     * @author Di Giorgio Domenico, Cris Malinconico
     * @param pManager
     *            the pool of currently running connections.
     */

    public ControlConnection(ConnectionPoolDataSource pManager) {
        ControlConnection.manager = pManager;
    }

    /**
     * Creates a new instance of the Thread.
     * 
     * @param pManager
     *            the pool of currently running connections.
     * @param pTime
     *            The time within which to re-establish the connection with mysql before
     *                che scada. This value must necessarily be less than
     *                value of the global wait_timeout variable of Mysql.
     */

    public ControlConnection(ConnectionPoolDataSource pManager, int pTime) {
        ControlConnection.waitTimeout = pTime;
        ControlConnection.manager = pManager;
    }

    /**
     * The thread does nothing but sleep when there are active users and
     *   keep the connection with MySQL open otherwise.
     * 
     */

    public void run() {
        try {
            while (true) {
                if (manager.activeSize() > 0) {
                    this.setPriority(Thread.MAX_PRIORITY);
                    manager.clearActive();
                    this.setPriority(Thread.NORM_PRIORITY);
                }
                if (manager.activeSize() == 0) {
                    while (true) {
                        try {
                            manager.closeAllPooledConnections();
                            Connection con = null;
                            con = manager.getConnection();
                            Statement st = con.createStatement();
                            st.executeQuery("show tables");
                            manager.release(con);
                            break;
                        } catch (Exception e) {
                            System.out.println("Generated exception "
                                    + "in Thread ControlConnection:" + e);
                        }
                    }
                    Thread.sleep(waitTimeout);
                } else {
                    Thread.sleep(waitTimeout);
                }
            }
        } catch (InterruptedException ex) {
            System.out.println("Thread ControlConnection aborted:" + ex);
        }
    }
}
