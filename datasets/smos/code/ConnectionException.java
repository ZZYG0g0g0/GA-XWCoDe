package smos.exception;

import java.lang.Exception;

/**
  * This class represents the exception generated when it is not possible
  * get a connection to the database
  */
public class ConnectionException extends Exception {
	
	private static final long serialVersionUID = -6593436034986073011L;

	/**
	 * Generate the exception without an associated error message.
	 */
	public ConnectionException() {
		super("Unable to Connect to the DataBase!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage 	The error message to be associated
	  *						with the exception.
	  */
	public ConnectionException(String pMessage) {
		super(pMessage);
	}
	
	
}