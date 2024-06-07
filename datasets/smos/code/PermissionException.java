package smos.exception;

import java.lang.Exception;

/**
  * This class represents the exception generated when a user
  * tries to perform an operation for which you do not have permission.
  */
public class PermissionException extends Exception {
	
	private static final long serialVersionUID = 1881009447251825664L;

	/**
	 * Generate the exception without an associated error message.
	 */
	public PermissionException() {
		super("Permission Denied!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage 	The error message to be associated
	  *						to the exception.
	  */
	public PermissionException(String pMessage) {
		super(pMessage);
	}
	
	
}