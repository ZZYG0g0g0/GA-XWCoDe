package smos.bean;

import smos.exception.InvalidValueException;

import java.io.Serializable;

/**
 *  Class used to model a user.
 *
 * 
 */
public class User implements Serializable{
	

	private static final long serialVersionUID = 7272532316912745508L;
	
	
	private int id;
	private String login;
	private String password;
	private String firstName;
	private String lastName;
	private int idParent;
	private String cell;
	private String eMail;
	
	/**
	 * The constructor of the class.
	 */
	public User(){
		this.id = 0 ;
	}
	
	
	/**
	 * @return Returns the user's login.
	 */
	public String getLogin() {
		return this.login;
	}
	
	/**
	 * Set the user login.
	 * param pLogin
	 * 			The login to be set.
	 * 
	 * @throws InvalidValueException 
	 */
	public void setLogin(String pLogin) throws InvalidValueException {
		if(pLogin.length()<=4)
			throw new InvalidValueException();
		else
			this.login = pLogin;
	}
	
	/**
	 * @return Returns the user's name.
	 */
	public String getName() {
		return this.lastName + " " + this.firstName;
	}
	
	/**
	 * @return Returns the user's name.
	 */
	public String getFirstName() {
		return this.firstName;
	}
	
	/**
	 * Set the user's name.
	 * @param pFirstName
	 * 			The name to set.
	 */
	public void setFirstName(String pFirstName) {
		this.firstName = pFirstName;
	}
	
	
	/**
	 * @return Returns the user's password.
	 */
	public String getPassword() {
		return this.password;
	}
	
	/**
	 * Set the user's password.
	 * @param pPassword
	 * 			The password to be set.
	 * 
	 * @throws InvalidValueException 
	 */
	public void setPassword(String pPassword) throws InvalidValueException {
		if(pPassword.length()<=4)
			throw new InvalidValueException();
		else
		this.password = pPassword;
	}
	
	/**
	 * @return Returns the user's surname.
	 */
	public String getLastName() {
		return this.lastName;
	}
	
	/**
	 * Set the user's last name.
	 * @param pLastName
	 * 			The surname to be set.
	 */
	public void setLastName(String pLastName) {
		this.lastName = pLastName;
	}
	
	/**
	 * return Returns the user's id.
	 */
	public int getId() {
		return this.id;
	}
	
	/**
	 * Set the user id.
	 * @param pId
	 * 			The id to be set.
	 */
	public void setId(int pId) {
		this.id = pId;
	}
	
	/**
	 * Returns a string containing the user's first and last name.
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		return this.getFirstName() 
		+ " " 
		+ this.getLastName();
	}
	
	/**
	 * @return the eMail
	 */
	public String getEMail() {
		return this.eMail;
	}
	/**
	 * @param pMail the eMail to set
	 */
	public void setEMail(String pMail) {
		this.eMail = pMail;
	}


	/**
	 * @return the cell
	 */
	public String getCell() {
		return this.cell;
	}


	/**
	 * @param cell the cell to set
	 */
	public void setCell(String pCell) {
		this.cell = pCell;
	}


	/**
	 * @return the idParent
	 */
	public int getIdParent() {
		return this.idParent;
	}


	/**
	 * @param idParent the idParent to set
	 */
	public void setIdParent(int pIdParent) {
		this.idParent = pIdParent;
	}
	
}
