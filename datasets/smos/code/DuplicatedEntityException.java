package smos.exception;

import java.lang.Exception;

/**
  * This class represents the exception generated when you
  * attempts to insert an entity already present in the database.
  */
public class DuplicatedEntityException extends Exception {
	
	private static final long serialVersionUID = 4858261134352455533L;

	/**
	 * Generate the exception without an associated error message.
	 */
	public DuplicatedEntityException() {
		super("Duplicate Key into the Repository!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage 	The error message to be associated
      *                     to the exception.
	  */
	public DuplicatedEntityException (String pMessage) {
		super(pMessage);
	}
	
	
}