package smos.exception;

import java.lang.Exception;

/**
  * This class represents the exception generated when a user
  * enter an incorrect password during authentication
  */
public class LoginException extends Exception {
	
	private static final long serialVersionUID = -1213284697567763493L;

	/**
	 * Generate the exception without an associated error message.
	 */
	public LoginException() {
		super("Login or Password Incorrect or Invalid Session!");
	}
	
	/**
	  * Generate the exception with an associated error message.
	  *
	  * @param pMessage 	The error message to be associated
      *                     to the exception.
	  */
	public LoginException(String pMessage) {
		super(pMessage);
	}
	
}