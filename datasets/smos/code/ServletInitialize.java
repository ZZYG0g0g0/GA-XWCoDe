package smos.application;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;

import smos.utility.Utility;

/**
 * Servlet used to initialize system parameters.
 * 
 * @author Bavota Gabriele, Carnevale Filomena.
 *
 */
public class ServletInitialize extends HttpServlet {

	private static final long serialVersionUID = -2542143445249797492L;
	
	@SuppressWarnings("unused")
	private ServletConfig config;
	
	 /**

     * Initialize the parameters

     */

    public void init(ServletConfig config) throws ServletException 

    {
    	this.config = config;
    	               
        
        //Set the smtp server specified in the xml configuration file
        Utility.setServerSmtp(config.getInitParameter("serverSmtp"));
        
        //Set the parameters needed to connect to the Database
        Utility.setDriverMySql(config.getInitParameter("driverMySql"));
        Utility.setFullPathDatabase(config.getInitParameter("fullPathDatabase"));
        Utility.setUserName(config.getInitParameter("userName"));
        Utility.setPassword(config.getInitParameter("password"));
        Utility.setMaxPoolSize(Integer.valueOf(config.getInitParameter("maxPoolSize")));
        Utility.setWaitTimeout(Integer.valueOf(config.getInitParameter("waitTimeout")));
        Utility.setActiveTimeout(Integer.valueOf(config.getInitParameter("activeTimeout")));
        Utility.setPoolTimeout(Integer.valueOf(config.getInitParameter("poolTimeout")));
        Utility.setTextFooter(config.getInitParameter("textFooter"));
        
        
	}

}
